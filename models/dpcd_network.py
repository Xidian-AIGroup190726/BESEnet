import torch
import torch.nn as nn
from torch.nn import init

from models.dpcd_modules import (
    BidirectionalSimilarityFusion,
    ChannelGroupedShuffleUnit,
    ConvNormActivation,
    CrossTemporalChannelExchange,
    DecoderStage,
    EncoderStage,
    ShapeFieldBoundaryHead,
    export_feature_maps,
)


class ProfessionalChangeDetectionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        feature_channels = [8, 16, 32, 64, 128]

        self.stem = nn.Sequential(
            ConvNormActivation(in_channel=3, out_channel=feature_channels[0], kernel=3, stride=1),
            ChannelGroupedShuffleUnit(in_channel=feature_channels[0]),
            ChannelGroupedShuffleUnit(in_channel=feature_channels[0]),
        )
        self.encoder_stage_2 = EncoderStage(in_channel=feature_channels[0], out_channel=feature_channels[1])
        self.encoder_stage_3 = EncoderStage(in_channel=feature_channels[1], out_channel=feature_channels[2])
        self.encoder_stage_4 = EncoderStage(in_channel=feature_channels[2], out_channel=feature_channels[3])
        self.encoder_stage_5 = EncoderStage(in_channel=feature_channels[3], out_channel=feature_channels[4])

        self.temporal_channel_exchange = CrossTemporalChannelExchange()
        self.boundary_projection = ShapeFieldBoundaryHead(in_channel=feature_channels[1])

        self.decoder_stage_1 = DecoderStage(in_channel=feature_channels[4], out_channel=feature_channels[3])
        self.decoder_stage_2 = DecoderStage(in_channel=feature_channels[3], out_channel=feature_channels[2])
        self.decoder_stage_3 = DecoderStage(in_channel=feature_channels[2], out_channel=feature_channels[1])

        self.semantic_fusion = nn.Conv2d(
            in_channels=feature_channels[1] * 2,
            out_channels=feature_channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.change_decoder_stage_4 = DecoderStage(in_channel=feature_channels[4], out_channel=feature_channels[3])
        self.change_decoder_stage_3 = DecoderStage(in_channel=feature_channels[3], out_channel=feature_channels[2])
        self.change_decoder_stage_2 = DecoderStage(in_channel=feature_channels[2], out_channel=feature_channels[1])

        self.t1_segmentation_head = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)
        self.t2_segmentation_head = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)
        self.shared_segmentation_head = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)

        self.similarity_fusion_1 = BidirectionalSimilarityFusion(in_channel=feature_channels[4])
        self.similarity_fusion_2 = BidirectionalSimilarityFusion(in_channel=feature_channels[3])
        self.similarity_fusion_3 = BidirectionalSimilarityFusion(in_channel=feature_channels[2])
        self.similarity_fusion_4 = BidirectionalSimilarityFusion(in_channel=feature_channels[1])

        self.t1_reconstruction_upsampler = self._build_output_upsampler(feature_channels)
        self.t2_reconstruction_upsampler = self._build_output_upsampler(feature_channels)
        self.shared_reconstruction_upsampler = self._build_output_upsampler(feature_channels)
        self.change_upsampler = self._build_output_upsampler(feature_channels)
        self.change_head = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

    @staticmethod
    def _build_output_upsampler(feature_channels):
        return nn.Sequential(
            nn.Conv2d(feature_channels[1], feature_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, t1_image, t2_image, log=False, img_name=None):
        t1_stage_1 = self.stem(t1_image)
        t2_stage_1 = self.stem(t2_image)

        if log:
            t1_stage_2 = self.encoder_stage_2(t1_stage_1, log=log, module_name="t1_encoder_stage_2", img_name=img_name)
            t2_stage_2 = self.encoder_stage_2(t2_stage_1, log=log, module_name="t2_encoder_stage_2", img_name=img_name)
            t1_stage_3 = self.encoder_stage_3(t1_stage_2, log=log, module_name="t1_encoder_stage_3", img_name=img_name)
            t2_stage_3 = self.encoder_stage_3(t2_stage_2, log=log, module_name="t2_encoder_stage_3", img_name=img_name)
            t1_stage_4 = self.encoder_stage_4(t1_stage_3, log=log, module_name="t1_encoder_stage_4", img_name=img_name)
            t2_stage_4 = self.encoder_stage_4(t2_stage_3, log=log, module_name="t2_encoder_stage_4", img_name=img_name)
            t1_stage_4, t2_stage_4 = self.temporal_channel_exchange(t1_stage_4, t2_stage_4)
            t1_stage_5 = self.encoder_stage_5(t1_stage_4, log=log, module_name="t1_encoder_stage_5", img_name=img_name)
            t2_stage_5 = self.encoder_stage_5(t2_stage_4, log=log, module_name="t2_encoder_stage_5", img_name=img_name)
        else:
            t1_stage_2 = self.encoder_stage_2(t1_stage_1)
            t2_stage_2 = self.encoder_stage_2(t2_stage_1)
            t1_stage_3 = self.encoder_stage_3(t1_stage_2)
            t2_stage_3 = self.encoder_stage_3(t2_stage_2)
            t1_stage_4 = self.encoder_stage_4(t1_stage_3)
            t2_stage_4 = self.encoder_stage_4(t2_stage_3)
            t1_stage_4, t2_stage_4 = self.temporal_channel_exchange(t1_stage_4, t2_stage_4)
            t1_stage_5 = self.encoder_stage_5(t1_stage_4)
            t2_stage_5 = self.encoder_stage_5(t2_stage_4)

        boundary_logits, _, _ = self.boundary_projection(torch.abs(t1_stage_2 - t2_stage_2))

        t1_decoder_4 = self.decoder_stage_1(t1_stage_5, t1_stage_4)
        t2_decoder_4 = self.decoder_stage_1(t2_stage_5, t2_stage_4)
        t1_decoder_3 = self.decoder_stage_2(t1_decoder_4, t1_stage_3)
        t2_decoder_3 = self.decoder_stage_2(t2_decoder_4, t2_stage_3)
        t1_decoder_2 = self.decoder_stage_3(t1_decoder_3, t1_stage_2)
        t2_decoder_2 = self.decoder_stage_3(t2_decoder_3, t2_stage_2)

        t1_segmentation_logits = self.t1_segmentation_head(self.t1_reconstruction_upsampler(t1_decoder_2))
        t2_segmentation_logits = self.t2_segmentation_head(self.t2_reconstruction_upsampler(t2_decoder_2))
        shared_segmentation = self.semantic_fusion(torch.cat([t1_decoder_2, t2_decoder_2], dim=1))
        shared_segmentation_logits = self.shared_segmentation_head(
            self.shared_reconstruction_upsampler(shared_segmentation)
        )

        if log:
            change_stage_5 = self.similarity_fusion_1(
                t1_stage_5, t2_stage_5, log=log, module_name="change_similarity_stage_5", img_name=img_name
            )
            change_stage_4 = self.change_decoder_stage_4(
                change_stage_5,
                self.similarity_fusion_2(
                    t1_decoder_4, t2_decoder_4, log=log, module_name="change_similarity_stage_4", img_name=img_name
                ),
            )
            change_stage_3 = self.change_decoder_stage_3(
                change_stage_4,
                self.similarity_fusion_3(
                    t1_decoder_3, t2_decoder_3, log=log, module_name="change_similarity_stage_3", img_name=img_name
                ),
            )
            change_stage_2 = self.change_decoder_stage_2(
                change_stage_3,
                self.similarity_fusion_4(
                    t1_decoder_2, t2_decoder_2, log=log, module_name="change_similarity_stage_2", img_name=img_name
                ),
            )
        else:
            change_stage_5 = self.similarity_fusion_1(t1_stage_5, t2_stage_5)
            change_stage_4 = self.change_decoder_stage_4(
                change_stage_5, self.similarity_fusion_2(t1_decoder_4, t2_decoder_4)
            )
            change_stage_3 = self.change_decoder_stage_3(
                change_stage_4, self.similarity_fusion_3(t1_decoder_3, t2_decoder_3)
            )
            change_stage_2 = self.change_decoder_stage_2(
                change_stage_3, self.similarity_fusion_4(t1_decoder_2, t2_decoder_2)
            )

        change_logits = self.change_head(self.change_upsampler(change_stage_2))

        if log:
            export_feature_maps(
                log_list=[
                    torch.sigmoid(change_logits),
                    t1_segmentation_logits,
                    t2_segmentation_logits,
                    shared_segmentation_logits,
                ],
                module_name="model",
                feature_name_list=["change_out", "seg_out1", "seg_out2", "seg_out_all"],
                img_name=img_name,
                module_output=False,
            )

        return (
            change_logits,
            t1_segmentation_logits,
            t2_segmentation_logits,
            shared_segmentation_logits,
            boundary_logits,
        )
