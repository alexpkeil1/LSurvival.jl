# Likelihood functions for time-to-event observations subject to left-truncation and right censoring

Define:

- $T$: Random variable for time at end of observation
- $E$: Random variable for time at beginning of observation
- $t_i$: time at study end for observation $i$
- $y_i$: event indicator (1=yes, 0=no) for observation $i$
- $e_i$: time at study entry for observation $i$
- $\mathbf{x}_i$: covariate vector for observation $i$
- $f(t)$: probability density function
- $S(t)=Pr(T>t)$: survival function where $S(0)= 1$
- $\mathscr{L} \equiv\mathscr{L}(t,e,y;\theta)$: likelihood function for observed data under a parametric distribution with parameters $\theta$

We then have that 
$$\begin{aligned}
\mathscr{L} =& \prod_i \mathscr{L}_i\\
\mathscr{L}_i =& \frac{f(t_i)^{y_i}S(t)^{(1-y_i)}}{S(e_i)}
\end{aligned}$$

Meaning that the likelihood (and log-likelihood) under a survival model need only have the probability density function and the survival function


### Weibull log-likelihood
The Weibull distribution can be parameterized as: (Kalbfleisch and Prentice, sec 2.2)
$$\begin{aligned} 
f(t)=&\lambda\gamma(\lambda t)^{\gamma-1}\exp(-(\lambda t)^{\gamma}) \\
S(t) =& \exp(-(\lambda t)^{\gamma})
\end{aligned}$$

Let $\gamma = \exp(-\rho), \lambda = \exp(-\alpha), z=\frac{\ln(t)-\alpha}{exp(\rho)}$ and using $t=\exp(\ln(t))$, we have that
$$\begin{aligned} 
f(t)=&\exp(-\alpha)\exp(-\rho)(\exp(-\alpha)\exp(\ln(t)))^{\exp(-\rho)-1}\exp(-(\exp(-\alpha) \exp(\ln(t)))^{\exp(-\rho)}) \\
f(t)=&\exp(-\alpha-\rho)\exp((\ln(t)-\alpha)(\exp(-\rho)-1))\exp(-(\exp(-\alpha) \exp(\ln(t))\exp(-\rho))) \\
\ln f(t)= & (-\alpha-\rho) + 
                 (\ln(t)-\alpha)(\exp(-\rho)-1) + 
                 -\exp(\ln(t)-\alpha)\exp(-\rho)) \\
= & (-\alpha-\rho) + 
                 (z-(\ln(t)-\alpha)) + 
                 -\exp(z) \\
= & z -\exp(z) -\rho -\ln(t)  \\
\end{aligned}$$


And 
$$\begin{aligned} 
S(t)=&\exp(-(exp(-\alpha) \exp(\ln(t)))^{\exp(-\rho)}) \\
S(t)=&\exp(-(\exp((\ln(t)-\alpha)\exp(-\rho))^{}) \\
S(t)=&\exp(-\exp(z)) \\
\ln S(t)=&-\exp(z) \\
\end{aligned}$$

So that the likelihood can be defined in terms of the natural log of the survival time, allowing the "location-scale" parameterization of a parametric survival model:

$$\begin{aligned} 
z_i =& \frac{\ln(t_i)-\mathbf{x}_i\beta}{\sigma}\\
\ln(t_i) =& \mathbf{x}_i\beta + \sigma z_i
\end{aligned}$$


Where the location parameter is given as $\alpha = \mathbf{x}\beta$ and the scale parmameter $\sigma=\exp(\rho)$ determines the magnitude of the error terms, whose distribution is $f(z)$ (e.g. in the case of the Weibull distribution for $t$, $w$ is the extreme value distribution). Here, the association between covariates $\mathbf{X}$ and the time-to-event outcome is characterized in terms of linear effects on the location parameter $\alpha$. Further details and interpretive assistance can be found in Kalbfleisch and Prentice.


### Exponential log-likelihood
This is a special case of the Weibull log-likelihood in which $\gamma=1$ (or, equivalently $\rho=0$)


### Log-normal log-likelihood

$$\begin{aligned} 
f(t)=&(2\pi)^{-1/2}\gamma t^{-1} \exp\bigg(\frac{-\gamma^2(\ln(\lambda t))^2}{2}   \bigg) \\
S(t) =& 1 - \Phi\big(\gamma \ln(\lambda t)\big)
\end{aligned}$$
