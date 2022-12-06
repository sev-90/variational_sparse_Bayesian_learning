# variational_sparse_Bayesian_learning
variational sparse Bayesian learning (relevance vector machine)
# Abstract
Accurate predictions of the route travel times and quantifying the reliability of the predictions are crucial in optimizing the service delivery transport in a city. This paper aims to predict the travel time distributions between any arbitrary locations in an urban network by training a probabilistic machine learning algorithm using historical trip data. In this project, variational relevance vector machines (VRVM) method and ensemble learning to probabilistically predict the trip travel time for any origin-destination pair at different times of day through learning the similarities between the previously observed travel times across the city road network. The similarities between the observed route travel times are quantified with multi-kernel function. Moreover, the VRVM method allows us to efficiently use historical data through sparse Bayesian learning that identifies the ``relevance" basis functions from the entire data.  

# Methodology
The variational relevance vector machine is a kernel-based probabilistic machine learning algorithm that is formulated based on the sparse Bayesian learning \cite{bishop2013variational, tipping2001sparse}. This algorithm has a functional form equivalent to the support vector machine while using a sparse set of basis functions gives higher generalization performance. In sparse Bayesian learning, most parameters are automatically estimated as zero, and only a few ``relevance'' parameters are non-zero. Therefore, this approach finds a set of relevant basis functions that can be used to make efficient predictions. Sparse Bayesian learning poses prior distributions over model parameters and hyperparameters and estimates their posteriors using optimization. While the posterior distributions of most irrelevant parameters become zero, the remaining parameters are relevant parameters for the predictions. In the variational relevance vector machine, variational inference approximates the optimal posterior distributions. The following section will summarize the Bayesian model structure and the variational posteriors.

## Model specification
Suppose our data is $\mathcal{D} = \{(\textbf{x}_i, y_i) | i=1,\dots, n\}$ where $\textbf{x}_i \in \mathbb{R}^d$ is the input feature vector, and $y_i \in \mathbb{R}$ is the target scalar. From historical data, our goal is to learn a model that given input vector $x_i$ outputs the value of the target variable $y_i$. In our travel time modeling context, using ambulance historical trip data, we train a probabilistic model that predicts the route travel time given origin-destination coordinations and the trip's start time. The probabilistic framework enables us to model the uncertainties in the ambulance travel times predictions by estimating both aleatoric and epistemic uncertainties that are attributed, to name a few, to the inherent stochasticity of the phenomenon, lack of enough knowledge, and imperfection model structure assumption. Based on the standard probabilistic formulation, the target value can be calculated by a function $f$ parametrized by $w$ as equation:

$$ y_i = f(x_i;w) + \epsilon_i $$

where to deal with travel time uncertainties, $\epsilon_i$s are assumed to be independent and identically distributed zero-mean Gaussian random noises, $\mathcal{N}(0,\lambda^{-1})$, with unkown variance $\lambda^{-1}$. This noise variance in the generative model captures the aleotric uncertainty in the travel time prediction. In probabailistic viewpoint, this means that the target variable, $y_i$, is generated by a Gaussian distribution that can be estimated by function $f(x_i;w)$ \eqref{y_dist} that in RVM it takes the form of the sum of linearly-weighted kernel functions $k(x,x_i)$ calculated over all pairs of $x_i$ and the relevance vectors $x$. Let $\Phi$ be the kernel matrix, $\Phi=[\phi(x_1),\dots,\phi(x_m)]$ and $\phi(x_i)=[1, d_i, k(x_i,x_1), \dots, k(x_i,x_m)]^T$, where $d_i,$ is the trip geodesic distance, and $m$ is the number of relevance vectors. Then, $y$ follows a Gaussian likelihood model written as

$$
\begin{aligned}
p(y|x)=\mathcal{N}(\Phi^Tw,\lambda^{-1})
\end{aligned}
%%
Here, we use Gaussian kernel to calculate multi-kernel functions since it is proven that Gaussian kernels yield higher prediction accuracy in machine learning application\cite{smola2004tutorial}. Kernel functions enable us to capture the non-linearity in the raw representation of the input space by mapping the input space onto the higher dimensional feature space that also gives rise to the higher computational power\cite{muller2001introduction}. Simply put, with kernel functions evaluated at each data point, we establish a linear relationship between the implicit feature space and the target values where the implicit higher dimensional feature space contains the quantified similarities between pairs of raw input features via Gaussian kernel function. The Gaussian multi-kernel function between two vectors $x_1$ and $x_2$ is expressed by

$$
\begin{aligned}
k(x_1,x_2)=\sum_{l\in L}\exp(-\frac{||x_1-x_2||^2}{\l^2}),
\end{aligned}

$$
where $L$ contains the width parameters that need to be specified a priori. The predictive accuracy of the Kernel-based models is highly influenced by the choice of width parameter, while finding its optimum value is a challenge. To identify the optimum width parameter, a general approach is to utilize the cross-validation technique. Besides, adopting the combination of multiple kernels with different width parameters instead of using one single kernel function can also increase the generalization of the kernel-based models. In our modeling setting, we adopt the combination of the multiple kernels, each constructed by a different width parameter. We find the optimum values for the width parameters using cross-validation.
Imposing a sparsity promoting distributions, zero-mean Gaussian prior over model parameters can control the complexity of the model and prevent overfitting. Thus, the unknown coefficient vector $w \in \mathbb{R}^m $ is assumed to have a multivariate Gaussian prior with the unknown variance $\sigma^2_w$ expressed by 

$$
\begin{aligned}
p(w) \sim \mathcal{N}(\textbf{0}, \sigma^2_w),\\
\end{aligned}
$$ 

We define $\sigma^2_w = (\lambda\Lambda)^{-1}$ where $\Lambda=\text{diag}(\alpha_1,...,\alpha_m)$. It implies that the parameter $w_i$ depends on $\alpha_i$ which corresponds to inverse of variance of $w_i$ \cite{ueda2002bayesian}.
In fully Bayesian paradigm, following the hierarchical prior setting, the variational RVM method considers hyperpriors over hyperparameters. To benefit from analytical propertries of the Gaussian distributions' conjugate priors, we pose a Gamma prior distribution over inverse noise variance, $\lambda$, as $p(\lambda)=\text{Gamma}(e_0,f_0)$ and a Gamma hyperprior distribution over $\alpha_k,k \in [1,\dots,m]$ as $p(\alpha_k)=\text{Gamma}(a_0,b_0)$. Initially, we set small values for $a_0$, $b_0$, $e_0$, and $f_0$ to have non-informative priors. This method called ``Automatic Relevance Determination (ARD)"  that automatically selects the relevant parameters $w_i$ based on the variance of the parameter $\sigma^2_{w_i}$ \cite{bishop2013variational}.