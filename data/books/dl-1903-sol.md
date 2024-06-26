UNIVERSITY OF STUTTGART
INSTITUTE OF SIGNAL PROCESSING AND SYSTEM THEORY
Prof. Dr.-Ing. B. Yang

| Written exam in   |            |
|-------------------|------------|
| Deep learning     | 11.02.2019 |
| 14:00–15:00 Uhr   | 3 Problems |

Solutions to Problem 1: Machine learning basics (14 points)
1.1 D (1P)
1.2 C (1P) 1.3 D (1P)
1.4 D (1P)
1.5 E (1P)

| 1.6   | A (1P)   |
|-------|----------|
| 1.7   | B (2P)   |

1.8 C (1P)

| 1.9   | E (1P)   |
|-------|----------|
| 1.10  | A (1P)   |
| 1.11  | B (1P)   |
| 1.12  | D (1P)   |
| 1.13  | C (1P)   |

2.1 B (1P)

| 2.2   | D (1P)   |
|-------|----------|

2.3 C (2P) 2.4 C (1P)
2.5 A (1P)
2.6 B (1P)
2.7 C (1P)
2.8 E (2P)
2.9 B (3P)
2.10 A (1P) 2.11 E (1P)
2.12 A (1P)
2.13 D (1P)
2.14 A (1P)
2.15 D (1P)
2.16 A (1P)
2.17 C (1P) 2.18 B (1P)
2.19 C (1P)
2.20 B (1P)
2.21 E (1P)
2.22 A (1P)
2 3.1 (3P) Due to

$$\begin{array}{r l}{p(x)}&{{}=}\\ {\ }&{}\\ {q(x)}&{{}=}\end{array}$$
p(x) = 1
$$\frac{1}{\sqrt{2\pi}\sigma_{1}}\exp\left(-\frac{(x-\mu_{1})^{2}}{2\sigma_{1}^{2}}\right),$$
q(x) = 1
we obtain

$$\ln\left(\frac{p(x)}{q(x)}\right)=\ln\left(\frac{\sigma_{2}}{\sigma_{1}}\right)-\frac{(x-\mu_{1})^{2}}{2\sigma_{1}^{2}}+\frac{(x-\mu_{2})^{2}}{2\sigma_{2}^{2}}.$$

3.2 (4P) Since

$$\begin{array}{r c l}{{E_{X\sim p}\left[\frac{(X-\mu_{1})^{2}}{2\sigma_{1}^{2}}\right]}}&{{=}}&{{\frac{\sigma_{1}^{2}}{2\sigma_{1}^{2}}=\frac{1}{2},}}\\ {{E_{X\sim p}\left[\frac{(X-\mu_{2})^{2}}{2\sigma_{2}^{2}}\right]}}&{{=}}&{{E_{X\sim p}\left[\frac{(X-\mu_{1}+\mu_{1}-\mu_{2})^{2}}{2\sigma_{2}^{2}}\right]=\frac{\sigma_{1}^{2}+(\mu_{1}-\mu_{2})^{2}}{2\sigma_{2}^{2}},}}\end{array}$$

we obtain

$$D_{\mathrm{KL}}(p||q)=\operatorname{E}_{X\sim p}\left[\ln\left({\frac{p(X)}{q(X)}}\right)\right]=\ln\left({\frac{\sigma_{2}}{\sigma_{1}}}\right)-{\frac{1}{2}}+{\frac{\sigma_{1}^{2}+(\mu_{1}-\mu_{2})^{2}}{2\sigma_{2}^{2}}}.$$

3.3 (2P) For i 6= j,

$${\frac{\partial x_{i}}{\partial a_{j}}}=-{\frac{e^{a_{i}}e^{a_{j}}}{(\sum_{k=1}^{c}e^{a_{k}})^{2}}}=-x_{i}x_{j}.$$

3.4 (2P)

$${\frac{\partial x_{i}}{\partial a_{i}}}={\frac{e^{a_{i}}(\sum_{k=1}^{c}e^{a_{k}})-e^{a_{i}}e^{a_{i}}}{(\sum_{k=1}^{c}e^{a_{k}})^{2}}}=x_{i}-x_{i}^{2}.$$

3.5 (2P) Hence the Jacobi matrix is

$${\frac{\partial{\underline{{x}}}}{\partial{\underline{{a}}}}}=\left[{\frac{\partial x_{i}}{\partial a_{j}}}\right]_{i j}={\left[\begin{array}{l l l}{x_{1}-x_{1}^{2}}&{-x_{1}x_{c}}\\ {}&{}&{}\\ {-x_{c}x_{1}}&{x_{c}-x_{c}^{2}}\end{array}\right]}=\operatorname{diag}({\underline{{x}}})-{\underline{{x}}}\,{\underline{{x}}}^{T}$$

where diag(x) is a diagonal matrix whose diagonal elements are those from x.

∇Lr(w) = ∇L(w) + 2λw.

3.7 (1P) If λ¯ = 0, there is no regularization and the original cost function L(w) is minimized via SGD.

3.8 (1P) If λ¯ = 0.5, w t+1 does not depend on w t and the SGD will fail.

3.9 (1P) If λ >¯ *0.5,* w t+1 follows −w tin a wrong direction. The SGD will fail.

3.10 (1P) λ < ¯ 0 will amplify w t. The result is an undesired weight increasing instead of weight decay.

3.11 (1P) The reasonable range for λ¯ is thus 0 < λ <¯ 0.5.