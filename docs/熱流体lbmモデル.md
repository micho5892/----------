---

# 熱流体格子BGKモデルと平衡分布関数に関する理論展開

## 1. 使用記号と変数の定義
本レポートにおける主要な変数の定義は以下の通りである。

*   $\rho$ : 流体の密度
*   $R$ : 気体定数
*   $T$ : 温度
*   $u, u_a$ : 流体の巨視的流速（ベクトル、成分表示）
*   $c_a, c_{pkia}$ : 粒子の離散速度（ベクトル）
*   $c_{pk}$ : 運動粒子の速度の大きさ ($c_{pk} = |c_{pkia}|$)
*   $c$ : 最小速度（計算上は $c=1$ とされることが多い）
*   $e$ : 内部エネルギー（温度に依存）
*   $P$ : 圧力
*   $f^{eq}$ : 連続的な局所平衡分布関数（Maxwell分布）
*   $f_a^{(0)}, f_{pki}^{(0)}$ : 離散化された局所平衡分布関数
*   $F_a, F_{pk}$ : 粒子の速さと内部エネルギーに依存する未定係数パラメータ
*   $B$ : 内部エネルギー $e$ の関数となる未定係数パラメータ
*   $a = (p, k, i)$ : 各種速度で運動する粒子を識別する指標集合
    *   $p$ : 運動粒子の種類（格子の種類）
    *   $k$ : 最小速度 $c$ の $k$ 倍の速さを持つ運動粒子を示す指標
    *   $i$ : 運動粒子の方向を示す指標（$i=0$ は静止粒子）
*   $\delta_{\alpha\beta}$ : クロネッカーのデルタ
*   $D$ : 次元数
*   $b$ : 運動方向数

---

## 2. 平衡分布関数の導出 (4.3.1節)

簡素化された局所平衡分布関数は、Maxwell分布 $f^{eq}$ を基礎にして導出される。連続的な変数に対するMaxwell分布は次のように定義される。

$$ f^{eq} = \frac{\rho}{(2\pi RT)^{\frac{3}{2}}} \exp\left[ - \frac{(c_a - u_a)^2}{2RT} \right] \quad (a=x,y,z) \quad \cdots (4.86) $$

格子Boltzmann法における離散的な粒子速度 $c_a$ に対する局所平衡分布関数を求めるため、式(4.86)を以下のように書き換える。

$$ f^{eq} = A \rho \exp\left[ B(c_a - u_a)^2 \right] \quad \cdots (4.87) $$

ここで、$A$ および $B$ はMaxwell分布の離散化により得られる分布関数を適切に決定するための未定係数である。

Mach数が低く、粒子の速さ $|c|$ に比べて流速の絶対値 $|u|$ が十分小さい流れを考慮し、式(4.87)を $u$ の3次の項までTaylor展開すると次式が得られる。

$$ f^{eq} \simeq A e^{B c^2} \rho \left[ 1 - 2B(c \cdot u) + 2B^2(c \cdot u)^2 + B(u \cdot u) - \frac{4}{3}B^3(c \cdot u)^3 - 2B^2(c \cdot u)(u \cdot u) \right] \quad \cdots (4.88) $$

さらに、式(4.88)で連続的な粒子速度を、空間格子に対応して離散化することにより、以下の局所平衡分布関数を得る。

$$ f_a^{(0)} = F_a \rho \left[ 1 - 2B(c_a \cdot u) + 2B^2(c_a \cdot u)^2 + B(u \cdot u) - \frac{4}{3}B^3(c_a \cdot u)^3 - 2B^2(c_a \cdot u)(u \cdot u) \right] \quad \cdots (4.89) $$

ここでパラメータ $F_a$ は、$A e^{Bc^2}$ を書き換えたものであり、粒子の速さと流体の内部エネルギー（温度）のみに依存する。もう一つのパラメータ $B$ は内部エネルギー $e$ の関数となる。
この近似式(4.89)は、内部エネルギー $e$ を含み流速の3次の項まで考慮している点で非圧縮性流体モデルとは異なる。

---

## 3. 3次元粒子分布モデル (3D 39V モデル) (4.3.2節)

3次元モデルとして、格子点に静止粒子のほかに26方向38種類の運動粒子が存在する「3D 39V モデル」を採用する。このモデルにおいて決定すべき未定係数は、$F_0, F_{11}, F_{12}, F_{13}, F_{22}, F_{31}$ および $B$ の計7個である。

これらを一意に決定するため、以下の1階から4階までの粒子速度モーメントの条件（等方性条件）を考慮する。

**奇数階（1階と3階）のモーメント:**
$$ \sum_i c_{pkia} = 0 \quad \cdots (4.90) $$
$$ \sum_i c_{pkia} c_{pki\beta} c_{pki\gamma} = 0 \quad \cdots (4.91) $$

**2階のモーメント:**
$$ \sum_i c_{pkia} c_{pki\beta} = \frac{b c_{pk}^2}{D} \delta_{\alpha\beta} \quad \cdots (4.92) $$

**4階のモーメント:**
$$ \sum_i c_{pkia} c_{pki\beta} c_{pki\gamma} c_{pki\delta} = 
\begin{cases} 
2(kc)^4 \delta_{\alpha\beta\gamma\delta} & (p=1, \ k=1,2,3) \\
4(kc)^4 (\Delta_{\alpha\beta\gamma\delta} - \delta_{\alpha\beta\gamma\delta}) & (p=2, \ k=2) \\
8(kc)^4 (\Delta_{\alpha\beta\gamma\delta} - 2\delta_{\alpha\beta\gamma\delta}) & (p=3, \ k=1) 
\end{cases} \quad \cdots (4.93) $$

ここで使用されるクロネッカーのデルタ記号等は以下の通り定義される。
$$ \delta_{\alpha\beta} = \begin{cases} 1 & (\alpha = \beta のとき) \\ 0 & (それ以外のとき) \end{cases} $$
$$ \delta_{\alpha\beta\gamma\delta} = \begin{cases} 1 & (\alpha = \beta = \gamma = \delta のとき) \\ 0 & (それ以外のとき) \end{cases} $$
$$ \Delta_{\alpha\beta\gamma\delta} = \delta_{\alpha\beta}\delta_{\gamma\delta} + \delta_{\alpha\gamma}\delta_{\beta\delta} + \delta_{\alpha\delta}\delta_{\beta\gamma} $$

4階テンソルが等方的であるための条件は以下となる。
$$ F_{11} + 16F_{12} + 81F_{13} - 32F_{22} - 8F_{31} = 0 \quad \cdots (4.97) $$

---

## 4. 未定係数の決定プロセスと最終式

巨視的な流れの支配方程式（質量、運動量、エネルギーの保存則およびNavier-Stokes方程式）を導出するための条件式と比較することで、7つの連立方程式が導かれ、係数が決定される。

計算の結果、パラメータ $B$ および圧力 $P$ は以下のように定まる。
$$ B = -\frac{3}{4e} \quad \cdots (4.112) $$
$$ P = \frac{2}{3}\rho e \quad \cdots (4.117) $$

これを踏まえ、その他の係数 $F_{pk}$ は以下のように決定される（$c=1$とした場合）。

$$ F_0 = 1 + \frac{1}{8Bc^2} \left( \frac{287}{80B^2c^4} + \frac{1549}{120Bc^2} + \frac{49}{3} \right) \quad \cdots (4.126) $$

$$ F_{11} = - \frac{1}{8Bc^2} \left( \frac{77}{80B^2c^4} + \frac{379}{120Bc^2} + 3 \right) \quad \cdots (4.127) $$

$$ F_{12} = \frac{1}{80Bc^2} \left( \frac{77}{40B^2c^4} + \frac{329}{60Bc^2} + 3 \right) \quad \cdots (4.128) $$

$$ F_{13} = - \frac{1}{120Bc^2} \left( \frac{21}{80B^2c^4} + \frac{67}{120Bc^2} + \frac{1}{3} \right) \quad \cdots (4.129) $$

$$ F_{22} = - \frac{1}{1280B^2c^4} \left( \frac{7}{2Bc^2} + 3 \right) \quad \cdots (4.130) $$

$$ F_{31} = \frac{1}{20B^2c^4} \left( \frac{7}{16Bc^2} + 1 \right) \quad \cdots (4.131) $$

このように決定された係数を有する局所平衡分布関数を用いることで、低Mach数流れの領域において、粒子分布関数の時間発展方程式がNavier-Stokes方程式系に支配される流れ場を表すことが保証される。

---

## 5. 補足：1次元粒子分布モデル (1D 7V モデル) (4.4.1節)

1次元空間における簡素化された局所平衡分布関数を用いたモデル（3種類の速さの運動粒子と静止粒子の計7種類、1D 7Vモデル）の場合、計5個の係数は3次元モデルと同様の方法で以下のように決定される。

$$ F_0 = 1 + \frac{1}{24Bc^2} \left( \frac{5}{4B^2c^4} + \frac{7}{Bc^2} + \frac{49}{3} \right) \quad \cdots (4.132) $$

$$ F_{11} = - \frac{1}{8Bc^2} \left( \frac{5}{16B^2c^4} + \frac{13}{8Bc^2} + 3 \right) \quad \cdots (4.133) $$

$$ F_{12} = \frac{1}{24Bc^2} \left( \frac{3}{8B^2c^4} + \frac{3}{2Bc^2} + \frac{9}{10} \right) \quad \cdots (4.134) $$

$$ F_{13} = - \frac{1}{144Bc^2} \left( \frac{3}{8B^2c^4} + \frac{3}{4Bc^2} + \frac{2}{5} \right) \quad \cdots (4.135) $$

$$ B = -\frac{1}{4e} \quad \cdots (4.136) $$

*(※注: 1次元モデルではパラメータ $B$ の係数が3次元の $-\frac{3}{4e}$ と異なり $-\frac{1}{4e}$ となる点に留意すること)*

---

# 熱流体格子BGKモデルと巨視的方程式の導出（続き）

## 6. 追加使用記号と変数の定義
前回の定義に加え、以下の展開で新たに使用される主要な記号の定義を示す。

*   $D$ : 空間の次元数
*   $\mathbf{c}_{pki}, \mathbf{c}_{\sigma i}$ : 2次元モデルにおける粒子速度ベクトル
*   $\sigma$ : 運動粒子の速さの倍率を示す指標（正6角格子モデルで使用）
*   $t_1, t_2$ : Chapman-Enskog展開における時間スケール変数
*   $r_{1\alpha}$ : Chapman-Enskog展開における空間スケール変数
*   $f_{pki}^{(1)}, f_{pki}^{(2)}$ : 分布関数の非平衡成分（それぞれ1次、2次の摂動項）
*   $\tau$ : 緩和時間
*   $\phi$ : 緩和に関連する無次元パラメータ
*   $P$ : 圧力
*   $\mu$ : 粘性係数
*   $\lambda$ : 第2粘性係数
*   $\kappa$ : 熱伝導係数
*   $c_s$ : 理論上の音速
*   $\gamma$ : 比熱比
*   $c_v, c_p$ : それぞれ定積比熱、定圧比熱
*   $P_r$ : Prandtl（プラントル）数
*   $\mathbf{U}_w$ : 壁面の移動速度
*   $e_w$ : 壁面温度から得られる内部エネルギー

---

## 7. 2次元粒子分布モデル (4.4.2節)

### 7.1 2次元正方格子モデル (2D 21V モデル)
2次元正方格子に基づくモデルにおいて、速度ベクトル $\mathbf{c}_{pki}$ は一般的に次のように表される。

$$ \mathbf{c}_{pki} = k\sqrt{p}c \left( \cos\left(\frac{\pi(i-1)}{2} + \frac{\pi(p-1)}{4}\right), \sin\left(\frac{\pi(i-1)}{2} + \frac{\pi(p-1)}{4}\right) \right) $$
$$ (i=1,\cdots,4, \quad p=1,2, \quad k=1,2,\cdots) \quad \cdots (4.137) $$

本モデル（2D 21V）における局所平衡分布関数の7個の未定係数は、質量、運動量、エネルギーの定義式およびエネルギー流束の関係式から以下のように求められる。

$$ F_0 = 1 + \frac{5}{4Bc^2} \left( \frac{17}{96B^2c^4} + \frac{35}{48Bc^2} + \frac{49}{45} \right) \quad \cdots (4.138) $$
$$ F_{11} = - \frac{1}{8Bc^2} \left( \frac{13}{16B^2c^4} + \frac{71}{24Bc^2} + 3 \right) \quad \cdots (4.139) $$
$$ F_{12} = \frac{1}{16Bc^2} \left( \frac{5}{16B^2c^4} + \frac{25}{24Bc^2} + \frac{3}{5} \right) \quad \cdots (4.140) $$
$$ F_{13} = - \frac{1}{24Bc^2} \left( \frac{1}{16B^2c^4} + \frac{1}{8Bc^2} + \frac{1}{15} \right) \quad \cdots (4.141) $$
$$ F_{21} = \frac{1}{4B^3c^6} \left( \frac{Bc^2}{3} + \frac{1}{8} \right) \quad \cdots (4.142) $$
$$ F_{22} = - \frac{1}{1536B^3c^6} (2Bc^2 + 3) \quad \cdots (4.143) $$
$$ B = - \frac{1}{2e} \quad \cdots (4.144) $$

### 7.2 正6角格子を用いたモデル (2D 19V モデル)
正6角格子上で $c, 2c, 3c$ の運動粒子が分布する 2D 19V (3-speed) モデルにおいて、粒子の速度ベクトル $\mathbf{c}_{\sigma i}$ は以下の式で表される。

$$ \mathbf{c}_{\sigma i} = \sigma c \left( \cos\left(\frac{\pi(i-1)}{3}\right), \sin\left(\frac{\pi(i-1)}{3}\right) \right) \quad (i=1,\cdots,6, \ \sigma=1,2,3) \quad \cdots (4.145) $$

このときの平衡分布関数は以下の形になる。
$$ f_{\sigma i}^{(0)} = F_{\sigma}\rho \left[ 1 - 2B \mathbf{c}_{\sigma i} \cdot \mathbf{u} + 2B^2(\mathbf{c}_{\sigma i} \cdot \mathbf{u})^2 + B(\mathbf{u} \cdot \mathbf{u}) - \frac{4}{3}B^3(\mathbf{c}_{\sigma i} \cdot \mathbf{u})^3 \right] \quad \cdots (4.146) $$

未定係数5個は同様の手法で以下のように決定される。
$$ F_0 = 1 + \frac{1}{36B^3c^6} (49B^2c^4 + 28Bc^2 + 6) \quad \cdots (4.147) $$
$$ F_1 = - \frac{1}{72B^3c^6} (18B^2c^4 + 13Bc^2 + 3) \quad \cdots (4.148) $$
$$ F_2 = \frac{1}{360B^3c^6} (9B^2c^4 + 20Bc^2 + 6) \quad \cdots (4.149) $$
$$ F_3 = - \frac{1}{1080B^3c^6} (2B^2c^4 + 5Bc^2 + 3) \quad \cdots (4.150) $$
$$ B = - \frac{1}{2e} \quad \cdots (4.151) $$

### 7.3 正6角格子 2-speed モデル (2D 13V) と高次モデル
同様に正6角格子 2D 13V モデルでは、速度の1次と2次の項からなる局所平衡分布関数が定義される。
$$ f_{\sigma i}^{(0)} = F_{\sigma}\rho \left[ 1 - 2B \mathbf{c}_{\sigma i \alpha} u_\alpha + 2B^2 c_{\sigma i \alpha} c_{\sigma i \beta} u_\alpha u_\beta + Bu^2 \right] \quad \cdots (4.152) $$

係数は以下の通りである。
$$ F_0 = 1 - \frac{1}{4B^2c^4}(2 + 5Bc^2) \quad \cdots (4.153) $$
$$ F_1 = - \frac{1}{9B^2c^4}(1 + 2Bc^2) \quad \cdots (4.154) $$
$$ F_2 = \frac{1}{36B^2c^4}\left(1 + \frac{1}{2}Bc^2\right) \quad \cdots (4.155) $$
$$ B = - \frac{1}{2e} \quad \cdots (4.156) $$

速度の3次の項まで考慮した3次オーダーモデルへ拡張する場合、平衡分布関数は以下のように記述される。
$$ f_{\sigma i}^{(0)} = F_{\sigma}\rho \left[ 1 - 2B \mathbf{c}_{\sigma i \alpha} u_\alpha + 2B^2 c_{\sigma i \alpha} c_{\sigma i \beta} u_\alpha u_\beta + Bu^2 + m c_{\sigma i \alpha} u_\alpha u^2 + n c_{\sigma i \alpha} c_{\sigma i \beta} c_{\sigma i \gamma} u_\alpha u_\beta u_\gamma \right] \quad \cdots (4.157) $$
$$ m = - \frac{F_1 + 16F_2}{108c^4F_1F_2} \quad \cdots (4.158a) $$
$$ n = \frac{F_1 + 4F_2}{81c^6F_1F_2} \quad \cdots (4.158b) $$

---

## 8. 巨視的な流れの支配方程式 (4.5節)

各次元における係数 $B$ は、一般的に $D$ 次元空間において以下のように表される。
$$ B = - \frac{D}{4e} \quad \cdots (4.159) $$

マルチスケール展開（Chapman-Enskog展開）を適用し、巨視的な時間および空間スケールで見た粒子集団の運動は以下のように表される。
$$ \left( \frac{\partial}{\partial t_1} + \frac{\partial}{\partial t_2} \right) f_{pki}^{(0)} + c_{pkia} \frac{\partial f_{pki}^{(0)}}{\partial r_{1\alpha}} + \left( 1 - \frac{1}{2\phi} \right) \left[ \frac{\partial f_{pki}^{(1)}}{\partial t_1} + c_{pkia} \frac{\partial f_{pki}^{(1)}}{\partial r_{1\alpha}} \right] = - \frac{1}{\tau\phi} (f_{pki}^{(1)} + f_{pki}^{(2)}) \quad \cdots (4.160) $$

### 8.1 連続の式
式(4.160)を $p, k, i$ について総和をとることで連続の式に関する関係が得られる。非平衡成分の和についての条件式（質量、運動量保存に関連）は以下の通りである。

$$ \sum_{p,k,i} f_{pki}^{(1)} = 0 \quad \cdots (4.162a), \quad \sum_{p,k,i} f_{pki}^{(2)} = 0 \quad \cdots (4.162b) $$
$$ \sum_{p,k,i} f_{pki}^{(1)} c_{pkia} = 0 \quad \cdots (4.163a), \quad \sum_{p,k,i} f_{pki}^{(2)} c_{pkia} = 0 \quad \cdots (4.163b) $$

これらを考慮すると、連続の式は以下のように導出される。
$$ \frac{\partial \rho}{\partial t} + \frac{\partial}{\partial r_{1\alpha}}(\rho u_\alpha) = 0 \quad \cdots (4.164) $$

### 8.2 運動方程式
式(4.160)の両辺に粒子移動速度 $c_{pki}$ を掛けて総和をとり、非平衡成分の条件を適用すると、Navier-Stokes方程式に相当する以下の巨視的運動方程式が得られる。

$$ \frac{\partial}{\partial t}(\rho u_\alpha) + \frac{\partial}{\partial r_{1\beta}}(\rho u_\alpha u_\beta) = - \frac{\partial P}{\partial r_{1\alpha}} + \frac{\partial}{\partial r_{1\beta}} \mu \left( \frac{\partial u_\beta}{\partial r_{1\alpha}} + \frac{\partial u_\alpha}{\partial r_{1\beta}} \right) + \frac{\partial}{\partial r_{1\alpha}} \left( \lambda \frac{\partial u_\gamma}{\partial r_{1\gamma}} \right) + G \quad \cdots (4.166) $$

ここで、$G$はMach数が大きくなければ無視できる高次の項である。圧力 $P$、粘性係数 $\mu$、第2粘性係数 $\lambda$ は $D$ 次元空間において以下で表される。
$$ P = \frac{2}{D}\rho e \quad \cdots (4.167) $$
$$ \mu = \frac{2}{D}\rho e \tau \left( \phi - \frac{1}{2} \right) \quad \cdots (4.168) $$
$$ \lambda = - \frac{4}{D^2}\rho e \tau \left( \phi - \frac{1}{2} \right) \quad \cdots (4.169) $$

### 8.3 エネルギー方程式と熱力学関係式
エネルギーの非平衡成分に関する条件式は以下の通りである。
$$ \sum_{p,k,i} \frac{1}{2} f_{pki}^{(1)} c_{pk}^2 = 0 \quad \cdots (4.170a), \quad \sum_{p,k,i} \frac{1}{2} f_{pki}^{(2)} c_{pk}^2 = 0 \quad \cdots (4.170b) $$

式(4.160)に $c_{pk}^2/2$ を乗じて総和をとり展開すると、エネルギー方程式が導出される。
$$ \frac{\partial}{\partial t}\left( \rho e + \frac{1}{2}\rho u^2 \right) + \frac{\partial}{\partial r_{1\alpha}} \left( \rho e + P + \frac{1}{2}\rho u^2 \right) u_\alpha = \frac{\partial}{\partial r_{1\alpha}} \left( \kappa' \frac{\partial e}{\partial r_{1\alpha}} \right) + \frac{\partial}{\partial r_{1\alpha}} \left\{ \mu u_\beta \left( \frac{\partial u_\beta}{\partial r_{1\alpha}} + \frac{\partial u_\alpha}{\partial r_{1\beta}} \right) \right\} + \frac{\partial}{\partial r_{1\alpha}} \left( \lambda \frac{\partial u_\beta}{\partial r_{1\beta}} u_\alpha \right) + H \quad \cdots (4.172) $$

ここで係数 $\kappa'$ は以下のように定義される。
$$ \kappa' = \frac{2(D+2)}{D^2} \tau \rho e \left( \phi - \frac{1}{2} \right) \quad \cdots (4.173) $$

理想気体の状態方程式 $P = R\rho T$ (4.174) との比較から、熱流体力学的性質を記述する各パラメータが以下のように定まる。
$$ RT = \frac{2}{D} e \quad \cdots (4.175) $$
熱伝導係数 $\kappa$:
$$ \kappa = \frac{DR}{2}\kappa' = \frac{(D+2)R}{D} \tau \rho e \left( \phi - \frac{1}{2} \right) \quad \cdots (4.176) $$
状態方程式の別表現と比熱比 $\gamma$:
$$ P = (\gamma - 1)\rho e \quad \cdots (4.177) $$
$$ \gamma = \frac{D+2}{D} \quad \cdots (4.178) $$
理論上の音速 $c_s$:
$$ c_s = \sqrt{\gamma \frac{P}{\rho}} = \sqrt{\frac{2(D+2)}{D^2}e} \quad \cdots (4.179) $$
定積比熱 $c_v$ および定圧比熱 $c_p$:
$$ c_v = \frac{D}{2}R \quad \cdots (4.181) $$
$$ c_p = \frac{D+2}{2}R \quad \cdots (4.182) $$

以上の関係から、本モデルにおける Prandtl数 $P_r$ は常に一定（$P_r = 1$）となることが示される。
$$ P_r = \frac{c_p \mu}{\kappa} = \frac{ \frac{D+2}{2}R \cdot \frac{2}{D}\rho e \tau \left( \phi - \frac{1}{2} \right) }{ \frac{D+2}{D} R \rho e \tau \left( \phi - \frac{1}{2} \right) } = 1 \quad \cdots (4.183) $$

---

## 9. 境界条件 (4.6節)

非圧縮性（非熱）流体流れの解析において用いられる、境界での粒子分布が「局所的平衡状態にある」と仮定した境界条件の設定方法は、平衡状態の粒子分布が巨視的な流れの状態から一意的に決定されるため扱いやすい。熱流体モデルにおいてもこの方法に従って境界条件を与えるが、非熱流体モデルでは粒子分布が「流体の密度」と「流速」から決められるのに対し、**熱流体モデルではそれに加えて「温度（内部エネルギー）」に関する条件が必要**になる。

### 9.1 固体壁境界における条件設定
熱流体格子BGKモデルにおける境界条件を、固体壁境界を例にとって説明する。
境界格子点上では、以下の手順で粒子分布を決定する。

1. **並進過程の処理**: 壁面を横切るような速度ベクトル $c_a$ を持つ運動粒子はとどまり、流体内部にある格子点へ向かう粒子のみが移動する。それと入れ替わりに、流体内部から新たな運動粒子が到着する。
2. **平衡分布関数の適用**: この状態で、流体の密度 $\rho$、流速 $u$、および温度（内部エネルギー $e$）で定義される平衡分布関数 $f_a^{(0)}$ に従って粒子分布を決定する。
    * $\rho$ : 格子点に存在する全粒子の密度
    * $u$ : 壁面の移動速度 $U_w$ に等しく設定する

内部エネルギー $e$ の与え方は、壁の熱的条件によって以下のように異なる。

**① 加熱あるいは冷却壁（等温壁）の条件:**
壁面温度から得られる内部エネルギーを $e_w$ とすると、次式となる。
$$ e = e_w \quad \cdots (4.184) $$

**② 断熱壁の条件:**
境界条件適用の前後で全粒子のエネルギーが保存されるように、次式を満たす数値を与える。
$$ \rho e = \sum_a \frac{1}{2} f_a (c_a - U_w)^2 \quad \cdots (4.185) $$
ここで、分布関数 $f_a$ は境界条件を適用する1時間ステップ前の粒子分布を意味している。

### 9.2 高速粒子に対する分割ステップ法
熱流体モデルでは、1度の並進過程で1格子点以上移動する高速な粒子が存在するため、壁近傍の格子点には1タイムステップ間に壁表面の位置を通過してしまうような運動粒子が存在し得る。「粒子が固体壁を通過しない」という原則を守るため、便宜上、**1並進過程（時間刻み $\tau$）を分割して複数の仮想的な時刻を設け、速度の異なる粒子が境界に到着する時刻ごとに境界条件を適用する**。

例えば、速さが $c, 2c, 3c$ の運動粒子分布 $f_{1i}, f_{2i}, f_{3i}$ を含む 2D 19V (3-speed) モデルの場合、1並進過程を4分割し、以下の手順で実行する。

1. 分割ステップ $t+\tau/3, t+\tau/2, t+2\tau/3$ において、それぞれ1格子点間距離だけ進む粒子（$f_{3i}$ や $f_{2i}$）についてのみ並進演算を行い、他の粒子はとどめたまま境界条件を実行する。
2. このとき、境界条件に用いられる平衡状態の密度は、その分割ステップに境界まで移動した粒子のみの総和により決定する。
3. 最後に時刻 $t+\tau$ となり、全運動粒子 $f_{1i}, f_{2i}, f_{3i}$ が同時に境界格子点に到着した後、4度目の境界条件を適用する。

### 9.3 滑りの境界条件
衝撃波管問題のような流れ場の計算において、境界層の影響を排除したい場合は**滑りの境界条件**を与える。この際、格子気体法でも用いられる「滑りの跳ね返り条件」を使用する。この境界条件の処理についても、上記と同様に1回の並進過程を分割する分割ステップを用いて実行される。