import torch
import torch.nn as nn

# リサイズサイズの設定
resize_size = 1024


# YOLOv8 バックボーン
class YOLOv8Backbone(nn.Module):
    def __init__(self, model_path):
        super(YOLOv8Backbone, self).__init__()
        # YOLOv8のロード
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = checkpoint['model'].model

        # スキップ接続を使用してFPNの特定のレイヤーを抽出
        self.stem = nn.Sequential(
            self.model[0],
            self.model[1],
            self.model[2]
        )
        self.dark3 = nn.Sequential(
            self.model[3],
            self.model[4]
        )
        self.dark4 = nn.Sequential(
            self.model[5],
            self.model[6]
        )
        self.dark5 = nn.Sequential(
            self.model[7],
            self.model[8]
        )

    def forward(self, x):
        # 入力とモデルの重みが同じ精度であることを確認
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        stem_out = self.stem(x)
        C3 = self.dark3(stem_out)
        C4 = self.dark4(C3)
        C5 = self.dark5(C4)
        return stem_out, C3, C4, C5


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.P5_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.P5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_1 = nn.Conv2d(128, 256, kernel_size=1)
        self.P4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.P4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_1 = nn.Conv2d(64, 256, kernel_size=1)
        self.P3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.P_stem_1 = nn.Conv2d(32, 256, kernel_size=1)
        self.P_stem_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upsample_P3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, stem_out, C3, C4, C5):
        P5 = self.P5_1(C5)
        P5_upsampled = self.P5_upsample(P5)

        P4 = self.P4_1(C4)
        P4 = self.P4_2(P4 + P5_upsampled)
        P4_upsampled = self.P4_upsample(P4)

        P3 = self.P3_1(C3)
        P3 = self.P3_2(P3 + P4_upsampled)
        P3_upsampled = self.upsample_P3(P3)

        P_stem = self.P_stem_1(stem_out)
        P_stem = self.P_stem_2(P_stem + P3_upsampled)

        return P_stem, P3, P4, P5


# ProtoNet
class ProtoNet(nn.Module):
    def __init__(self, num_prototypes):
        super(ProtoNet, self).__init__()
        self.protonet = nn.Sequential(
            nn.Conv2d(256, num_prototypes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_prototypes, num_prototypes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.protonet(x)


# Detect Branch
class DetectBranch(nn.Module):
    def __init__(self, num_classes):
        super(DetectBranch, self).__init__()
        self.detect = nn.Conv2d(256, num_classes, kernel_size=1)
        self.mask_coeff = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.detect(x), self.mask_coeff(x)


# Mask Branch
class MaskBranch(nn.Module):
    def __init__(self, num_prototypes, num_classes):
        super(MaskBranch, self).__init__()
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv1x1 = nn.Conv2d(num_prototypes, 1, kernel_size=1)

    def forward(self, prototypes, mask_coeffs):
        batch_size = prototypes.size(0)
        height, width = prototypes.size(2), prototypes.size(3)

        # 各クラスごとのマスクを計算
        masks = []
        for i in range(self.num_classes):
            mask = torch.sum(prototypes * mask_coeffs[:, i, :, :].unsqueeze(1), dim=1)
            masks.append(mask)

        masks = torch.stack(masks, dim=1)
        masks = masks.view(batch_size * self.num_classes, 1, height, width)
        masks = self.upsample(masks)

        masks = masks.view(batch_size, self.num_classes, height * 4, width * 4)
        return masks


# FastFPN
class FastFPN(nn.Module):
    def __init__(self, num_classes=1, num_prototypes=8):
        super(FastFPN, self).__init__()
        self.backbone = YOLOv8Backbone('yolov8n.pt')
        self.fpn = FPN()
        self.protonet = ProtoNet(num_prototypes)
        self.detect_branch = DetectBranch(num_classes)
        self.mask_branch = MaskBranch(num_prototypes, num_classes)

    def forward(self, x):
        stem_out, C3, C4, C5 = self.backbone(x)
        P_stem, P3, P4, P5 = self.fpn(stem_out, C3, C4, C5)
        prototypes = self.protonet(P_stem)
        detections, mask_coeffs = self.detect_branch(P_stem)
        masks = self.mask_branch(prototypes, mask_coeffs)

        return detections, masks
