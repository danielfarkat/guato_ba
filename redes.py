#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
guato_ba · redes.py
Todas as arquiteturas de aprendizado do estudo (self-contained, prontas p/ git):

Deep learning (6):
  unet          — U-Net simples sobre concat(t1,t2)            [in_ch=8, bandas red,b8a,b11,b12]
  att_unet      — U-Net + portões de atenção (Oktay)           [in_ch=8]
  biunet        — encoder siamês (pesos compartilhados), sem atenção [in_ch=6, bandas b8a,b11,b12]
  biatt_unet    — Bi-RAUnet: concat + atenção + bloco residual [in_ch=6]
  resnet34_siam — encoder ResNet34 ImageNet siamês + decoder U-Net [in_ch=6]
  deeplabv3plus — backbone residual + ASPP multiescala + decoder [in_ch=8]

Machine learning (4): rf, xgboost, mlp, svm (fábricas).
Loss: Focal Tversky (dominante) + BCE ponderada por borda.
Entrada dos modelos JÁ vem sem água/nuvem (SCL+WIW aplicados a montante; inválidos zerados).
Saída dos DL: logits (B,H,W).
"""
import os, torch, torch.nn as nn, torch.nn.functional as F
_NORM = os.environ.get("GUATO_NORM", "bn")   # "bn" (default=BatchNorm) ou "gn" (GroupNorm: sem gap train/eval, robusto a batch pequeno)
def _gn(c):
    if _NORM == "gn":
        g = next((x for x in (32, 16, 8, 4, 2, 1) if c % x == 0), 1)
        return nn.GroupNorm(g, c)
    return nn.BatchNorm2d(c)

def _act(): return nn.ReLU(inplace=True)

# ============================================================
#  Blocos base
# ============================================================
class ConvBNAct(nn.Module):
    def __init__(s, ic, oc, k=3):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(ic, oc, k, padding=k // 2, bias=False), _gn(oc), _act())
    def forward(s, x): return s.net(x)

class DoubleConv(nn.Module):
    def __init__(s, ic, oc):
        super().__init__(); s.net = nn.Sequential(ConvBNAct(ic, oc), ConvBNAct(oc, oc))
    def forward(s, x): return s.net(x)

class ResidualBlock(nn.Module):
    def __init__(s, ic, oc):
        super().__init__()
        s.c1 = nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1, bias=False), _gn(oc), _act())
        s.c2 = nn.Sequential(nn.Conv2d(oc, oc, 3, padding=1, bias=False), _gn(oc))
        s.proj = nn.Conv2d(ic, oc, 1, bias=False) if ic != oc else nn.Identity(); s.out = _act()
    def forward(s, x): return s.out(s.c2(s.c1(x)) + s.proj(x))

class AttentionGate(nn.Module):
    """x=skip (alta res); g=gating (baixa res). Oktay 2018."""
    def __init__(s, ch_x, ch_g, ch_int):
        super().__init__()
        s.wx = nn.Sequential(nn.Conv2d(ch_x, ch_int, 1, stride=2, bias=True), _gn(ch_int))
        s.wg = nn.Sequential(nn.Conv2d(ch_g, ch_int, 1, stride=1, bias=True), _gn(ch_int))
        s.psi = nn.Sequential(nn.Conv2d(ch_int, 1, 1, bias=True), _gn(1), nn.Sigmoid())
        s.relu = _act()
    def forward(s, x, g):
        a = s.psi(s.relu(s.wx(x) + s.wg(g)))
        a = F.interpolate(a, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x * a

# ============================================================
#  1) UNet simples  (concat t1|t2)
# ============================================================
class UNet(nn.Module):
    def __init__(s, in_ch=8, f=48, p=0.0):
        super().__init__()
        s.e1 = DoubleConv(in_ch, f); s.e2 = DoubleConv(f, f*2); s.e3 = DoubleConv(f*2, f*4); s.e4 = DoubleConv(f*4, f*8)
        s.pool = nn.MaxPool2d(2); s.bn = DoubleConv(f*8, f*16); s.drop = nn.Dropout2d(p)
        s.u4 = nn.ConvTranspose2d(f*16, f*8, 2, 2); s.d4 = DoubleConv(f*16, f*8)
        s.u3 = nn.ConvTranspose2d(f*8, f*4, 2, 2);  s.d3 = DoubleConv(f*8, f*4)
        s.u2 = nn.ConvTranspose2d(f*4, f*2, 2, 2);  s.d2 = DoubleConv(f*4, f*2)
        s.u1 = nn.ConvTranspose2d(f*2, f, 2, 2);    s.d1 = DoubleConv(f*2, f)
        s.out = nn.Conv2d(f, 1, 1)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2)); e4 = s.e4(s.pool(e3))
        b = s.drop(s.bn(s.pool(e4)))
        d = s.d4(torch.cat([s.u4(b), e4], 1)); d = s.d3(torch.cat([s.u3(d), e3], 1))
        d = s.d2(torch.cat([s.u2(d), e2], 1)); d = s.d1(torch.cat([s.u1(d), e1], 1))
        return s.out(d).squeeze(1)

# ============================================================
#  2) Attention UNet  (concat + portões de atenção)
# ============================================================
class AttUNet(nn.Module):
    def __init__(s, in_ch=8, f=48, p=0.0):
        super().__init__()
        s.e1 = DoubleConv(in_ch, f); s.e2 = DoubleConv(f, f*2); s.e3 = DoubleConv(f*2, f*4); s.e4 = DoubleConv(f*4, f*8)
        s.pool = nn.MaxPool2d(2); s.bn = DoubleConv(f*8, f*16); s.drop = nn.Dropout2d(p)
        s.a4 = AttentionGate(f*8, f*16, f*8); s.u4 = nn.ConvTranspose2d(f*16, f*8, 2, 2); s.d4 = DoubleConv(f*16, f*8)
        s.a3 = AttentionGate(f*4, f*8, f*4);  s.u3 = nn.ConvTranspose2d(f*8, f*4, 2, 2);  s.d3 = DoubleConv(f*8, f*4)
        s.a2 = AttentionGate(f*2, f*4, f*2);  s.u2 = nn.ConvTranspose2d(f*4, f*2, 2, 2);  s.d2 = DoubleConv(f*4, f*2)
        s.a1 = AttentionGate(f, f*2, f);      s.u1 = nn.ConvTranspose2d(f*2, f, 2, 2);    s.d1 = DoubleConv(f*2, f)
        s.out = nn.Conv2d(f, 1, 1)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2)); e4 = s.e4(s.pool(e3))
        b = s.drop(s.bn(s.pool(e4)))
        d = s.d4(torch.cat([s.u4(b), s.a4(e4, b)], 1))
        d = s.d3(torch.cat([s.u3(d), s.a3(e3, d)], 1))
        d = s.d2(torch.cat([s.u2(d), s.a2(e2, d)], 1))
        d = s.d1(torch.cat([s.u1(d), s.a1(e1, d)], 1))
        return s.out(d).squeeze(1)

# ============================================================
#  3) BiUNet  (encoder siamês compartilhado, sem atenção)
# ============================================================
class BiUNet(nn.Module):
    def __init__(s, in_ch=6, f=48, p=0.0):
        super().__init__()
        c = in_ch // 2                     # 3 bandas por data
        s.e1 = DoubleConv(c, f); s.e2 = DoubleConv(f, f*2); s.e3 = DoubleConv(f*2, f*4); s.e4 = DoubleConv(f*4, f*8)
        s.pool = nn.MaxPool2d(2); s.bn = DoubleConv(f*8, f*16); s.drop = nn.Dropout2d(p)
        s.u4 = nn.ConvTranspose2d(f*16*2, f*8, 2, 2); s.d4 = DoubleConv(f*8 + f*8*2, f*8)
        s.u3 = nn.ConvTranspose2d(f*8, f*4, 2, 2);    s.d3 = DoubleConv(f*4 + f*4*2, f*4)
        s.u2 = nn.ConvTranspose2d(f*4, f*2, 2, 2);    s.d2 = DoubleConv(f*2 + f*2*2, f*2)
        s.u1 = nn.ConvTranspose2d(f*2, f, 2, 2);      s.d1 = DoubleConv(f + f*2, f)
        s.out = nn.Conv2d(f, 1, 1)
    def _enc(s, x):
        e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2)); e4 = s.e4(s.pool(e3))
        b = s.bn(s.pool(e4)); return e1, e2, e3, e4, b
    def forward(s, x):
        c = x.shape[1] // 2
        a1, a2, a3, a4, ab = s._enc(x[:, :c]); b1, b2, b3, b4, bb = s._enc(x[:, c:])
        b = s.drop(torch.cat([ab, bb], 1))
        d = s.d4(torch.cat([s.u4(b),  a4, b4], 1))
        d = s.d3(torch.cat([s.u3(d), a3, b3], 1))
        d = s.d2(torch.cat([s.u2(d), a2, b2], 1))
        d = s.d1(torch.cat([s.u1(d), a1, b1], 1))
        return s.out(d).squeeze(1)

# ============================================================
#  4) BiAttUNet = Bi-RAUnet (concat + atenção + residual)
# ============================================================
class BiAttUNet(nn.Module):
    def __init__(s, in_ch=6, base=48, p=0.0):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, base*8; c5 = base*16
        s.e1 = DoubleConv(in_ch, c1); s.e2 = DoubleConv(c1, c2); s.e3 = DoubleConv(c2, c3); s.e4 = DoubleConv(c3, c4)
        s.pool = nn.MaxPool2d(2); s.bottleneck = DoubleConv(c4, c5); s.drop = nn.Dropout2d(p)
        s.att4 = AttentionGate(c4, c5, c4); s.up4 = nn.ConvTranspose2d(c5, c5, 2, 2); s.res4 = ResidualBlock(c5+c4, c4)
        s.att3 = AttentionGate(c3, c4, c3); s.up3 = nn.ConvTranspose2d(c4, c4, 2, 2); s.res3 = ResidualBlock(c4+c3, c3)
        s.att2 = AttentionGate(c2, c3, c2); s.up2 = nn.ConvTranspose2d(c3, c3, 2, 2); s.res2 = ResidualBlock(c3+c2, c2)
        s.att1 = AttentionGate(c1, c2, c1); s.up1 = nn.ConvTranspose2d(c2, c2, 2, 2); s.res1 = ResidualBlock(c2+c1, c1)
        s.head = nn.Conv2d(c1, 1, 1)
    def forward(s, x):
        c1 = s.e1(x); c2 = s.e2(s.pool(c1)); c3 = s.e3(s.pool(c2)); c4 = s.e4(s.pool(c3))
        c5 = s.drop(s.bottleneck(s.pool(c4)))
        d = s.res4(torch.cat([s.up4(c5), s.att4(c4, c5)], 1))
        d = s.res3(torch.cat([s.up3(d),  s.att3(c3, d)], 1))
        d = s.drop(s.res2(torch.cat([s.up2(d), s.att2(c2, d)], 1)))
        d = s.res1(torch.cat([s.up1(d),  s.att1(c1, d)], 1))
        return s.head(d).squeeze(1)

# ============================================================
#  5) ResNet34 Siamese UNet (ImageNet)
# ============================================================
class ResNet34SiamUNet(nn.Module):
    def __init__(s, in_ch=6, pretrained=True, p=0.0):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights
        w = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        s.enc = resnet34(weights=w)
        s.drop = nn.Dropout2d(p)
        s.up4 = nn.ConvTranspose2d(512*2, 256, 2, 2); s.d4 = DoubleConv(256 + 256*2, 256)
        s.up3 = nn.ConvTranspose2d(256, 128, 2, 2);   s.d3 = DoubleConv(128 + 128*2, 128)
        s.up2 = nn.ConvTranspose2d(128, 64, 2, 2);    s.d2 = DoubleConv(64 + 64*2, 64)
        s.up1 = nn.ConvTranspose2d(64, 64, 2, 2);     s.d1 = DoubleConv(64 + 64*2, 64)
        s.up0 = nn.ConvTranspose2d(64, 32, 2, 2);     s.d0 = DoubleConv(32, 32)
        s.head = nn.Conv2d(32, 1, 1)
    def _feats(s, x):
        e = s.enc
        x0 = e.relu(e.bn1(e.conv1(x)))      # /2, 64
        x1 = e.layer1(e.maxpool(x0))        # /4, 64
        x2 = e.layer2(x1)                   # /8, 128
        x3 = e.layer3(x2)                   # /16, 256
        x4 = e.layer4(x3)                   # /32, 512
        return x0, x1, x2, x3, x4
    def forward(s, x):
        x1, x2 = x[:, :3], x[:, 3:]
        a0, a1, a2, a3, a4 = s._feats(x1); b0, b1, b2, b3, b4 = s._feats(x2)
        d = s.d4(torch.cat([s.up4(s.drop(torch.cat([a4, b4], 1))), a3, b3], 1))
        d = s.d3(torch.cat([s.up3(d), a2, b2], 1))
        d = s.d2(torch.cat([s.up2(d), a1, b1], 1))
        d = s.d1(torch.cat([s.up1(d), a0, b0], 1))
        d = s.d0(s.up0(d))
        return s.head(d).squeeze(1)

# ============================================================
#  6) DeepLabV3+ (backbone residual + ASPP + decoder), concat bitemporal
# ============================================================
class _ResStage(nn.Module):
    def __init__(s, ic, oc, stride=1):
        super().__init__()
        s.b1 = ResidualBlock(ic, oc)
        s.down = nn.Sequential(nn.Conv2d(oc, oc, 3, stride=stride, padding=1, bias=False), _gn(oc), _act()) if stride > 1 else nn.Identity()
        s.b2 = ResidualBlock(oc, oc)
    def forward(s, x): return s.b2(s.down(s.b1(x)))

class _ASPP(nn.Module):
    def __init__(s, ic, oc=256, rates=(1, 6, 12, 18), p=0.0):
        super().__init__()
        s.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ic, oc, 1 if r == 1 else 3, padding=0 if r == 1 else r, dilation=r, bias=False),
                          _gn(oc), _act()) for r in rates])
        s.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ic, oc, 1, bias=False), _gn(oc), _act())
        s.proj = nn.Sequential(nn.Conv2d(oc * (len(rates) + 1), oc, 1, bias=False), _gn(oc), _act(), nn.Dropout2d(p))
    def forward(s, x):
        feats = [b(x) for b in s.branches]
        g = s.pool(x); g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        return s.proj(torch.cat(feats + [g], 1))

class DeepLabV3Plus(nn.Module):
    def __init__(s, in_ch=8, f=48, p=0.0):
        super().__init__()
        s.stem = DoubleConv(in_ch, f)                 # /1
        s.s1 = _ResStage(f, f*2, stride=2)            # /2  (low-level)
        s.s2 = _ResStage(f*2, f*4, stride=2)          # /4
        s.s3 = _ResStage(f*4, f*8, stride=2)          # /8
        s.s4 = _ResStage(f*8, f*8, stride=2)          # /16 (encoder final)
        s.aspp = _ASPP(f*8, 256, p=p)
        s.low_proj = nn.Sequential(nn.Conv2d(f*2, 48, 1, bias=False), _gn(48), _act())
        s.dec = nn.Sequential(DoubleConv(256 + 48, 256), nn.Dropout2d(p), nn.Conv2d(256, 1, 1))
    def forward(s, x):
        x0 = s.stem(x); low = s.s1(x0); e = s.s4(s.s3(s.s2(low)))      # low=/2, e=/16
        a = s.aspp(e)
        a = F.interpolate(a, size=low.shape[2:], mode="bilinear", align_corners=False)
        d = s.dec(torch.cat([a, s.low_proj(low)], 1))
        return F.interpolate(d, size=x.shape[2:], mode="bilinear", align_corners=False).squeeze(1)

# ============================================================
#  Registro dos modelos DL
# ============================================================
MODELOS_DL = {
    "unet":          dict(cls=UNet,            in_ch=8, bandas=[0, 1, 2, 3], micro=256),
    "unet_nored":    dict(cls=UNet,            in_ch=6, bandas=[1, 2, 3],    micro=256),  # = unet sem o RED (isola o efeito do vermelho/nuvem)
    "att_unet":      dict(cls=AttUNet,         in_ch=8, bandas=[0, 1, 2, 3], micro=128),
    "biunet":        dict(cls=BiUNet,          in_ch=6, bandas=[1, 2, 3],    micro=128),
    "biatt_unet":    dict(cls=BiAttUNet,       in_ch=6, bandas=[1, 2, 3],    micro=128),
    "resnet34_siam": dict(cls=ResNet34SiamUNet,in_ch=6, bandas=[1, 2, 3],    micro=64),
    "deeplabv3plus": dict(cls=DeepLabV3Plus,   in_ch=8, bandas=[0, 1, 2, 3], micro=96),
}

def construir_dl(nome, p=0.2, pretrained=True):
    cfg = MODELOS_DL[nome]; cls = cfg["cls"]
    if nome == "resnet34_siam":
        return cls(in_ch=cfg["in_ch"], pretrained=pretrained, p=p)
    return cls(in_ch=cfg["in_ch"], p=p)

# ============================================================
#  Loss: Focal Tversky (dominante) + BCE ponderada por borda
# ============================================================
def make_loss(border_w=4.0, tv_a=0.3, tv_b=0.7, tv_g=0.75, w_bce=0.5, ign=255):
    def loss_fn(logit, target, pw):
        valid = (target != ign); t = target.clamp(0, 1).float()
        if valid.sum() == 0:
            return logit.sum() * 0.0
        with torch.no_grad():
            tt = t.unsqueeze(1)
            band = ((F.max_pool2d(tt, 3, 1, 1) + F.max_pool2d(-tt, 3, 1, 1)) > 0).float().squeeze(1)
            w = 1.0 + border_w * band
        bce = (F.binary_cross_entropy_with_logits(logit, t, pos_weight=pw, reduction="none") * w)[valid].mean()
        p = torch.sigmoid(logit[valid]); tv = t[valid]
        TP = (p * tv).sum(); FP = (p * (1 - tv)).sum(); FN = ((1 - p) * tv).sum()
        TI = (TP + 1) / (TP + tv_a * FP + tv_b * FN + 1); ftl = (1 - TI) ** tv_g
        return ftl + w_bce * bce
    return loss_fn

# ============================================================
#  Machine learning (4) — fábricas
# ============================================================
def make_rf(**hp):
    from sklearn.ensemble import RandomForestClassifier
    d = dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features="sqrt",
             n_jobs=4, class_weight="balanced", random_state=42); d.update(hp)
    return RandomForestClassifier(**d)

def make_xgb(**hp):
    from xgboost import XGBClassifier
    d = dict(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
             tree_method="hist", n_jobs=4, eval_metric="logloss", random_state=42); d.update(hp)
    return XGBClassifier(**d)

def make_mlp(**hp):
    from sklearn.neural_network import MLPClassifier
    d = dict(hidden_layer_sizes=(128, 64), alpha=1e-4, learning_rate_init=1e-3, max_iter=200,
             early_stopping=True, random_state=42); d.update(hp)
    return MLPClassifier(**d)

def make_svm(**hp):
    from sklearn.svm import SVC
    d = dict(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42); d.update(hp)
    return SVC(**d)

MODELOS_ML = {"rf": make_rf, "xgboost": make_xgb, "mlp": make_mlp, "svm": make_svm}

if __name__ == "__main__":
    import numpy as np
    x8 = torch.randn(2, 8, 128, 128); x6 = torch.randn(2, 6, 128, 128)
    for nm in MODELOS_DL:
        m = construir_dl(nm, p=0.2, pretrained=False)
        x = x8 if MODELOS_DL[nm]["in_ch"] == 8 else x6
        y = m(x)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{nm:14s} out={tuple(y.shape)}  params={n:.2f}M")
