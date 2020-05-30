# [Biologically-informed neural networks](https://arxiv.org/abs/2005.13073)

![](https://github.com/jlager/BINNs/blob/master/Figures/schematic.png?raw=true)

Biologically-informed neural networks (BINNs), an extension of physics-informed neural networks [\[1\]](https://www.sciencedirect.com/science/article/pii/S0021999118307125), are introduced and used to discover the underlying dynamics of biological systems from sparse experimental data. BINNs are trained in a supervised learning framework to approximate *in vitro* cell biology assay experiments while respecting a generalized form of the governing reaction-diffusion partial differential equation (PDE). By allowing the diffusion and reaction terms to be multilayer perceptrons (MLPs), the nonlinear forms of these terms can be learned while simultaneously converging to the solution of the governing PDE. Further, the trained MLPs are used to guide the selection of biologically interpretable mechanistic forms of the PDE terms which provides new insights into the biological and physical mechanisms that govern the dynamics of the observed system. The method is evaluated on sparse real-world data from wound healing assays with varying initial cell densities [\[2\]](https://www.sciencedirect.com/science/article/abs/pii/S0022519315005676#f0005).

For more information, see the associated [manuscript](https://arxiv.org/abs/2005.13073).

All of the code is implemented using Python (available [here](https://www.anaconda.com/products/individual)) and the open source machine learning framework, PyTorch (available [here](https://pytorch.org/get-started/locally/)).

The **Notebooks** folder contains the Jupyter Notebooks used to train and evaluate BINNs. To train BINNs, see `BINNsTraining.ipynb`.To reproduce the figures in the manuscript, the BINNs-related figures corresponding to the *experimental* scratch assay data (i.e. Figures 3-7 and Supplementary Figures S4, S5, S8, and S9) are included in `BINNsEvaluation.ipynb`, BINNs-related figures corresponding to the *simulated* scratch assay data (i.e. Supplementary Figures S2 and S3) are included in `SimulationCaseStudy.ipynb`, and mechanistic modeling results (i.e. Figure 8 and Supplementary Figures S6 and S7) are included in `MechanisticModeling.ipynb`. These figures are also accessible in the **Figures** folder.

The experimental cell migration data from [\[2\]](https://www.sciencedirect.com/science/article/abs/pii/S0022519315005676#f0005) and the classical Fisher-KPP and Generalized Porous-Fisher-KPP simulation data are included in the **Data** folder.

The parameters (i.e. weights and biases) of the neural network models comprising BINNs are included in the **Weights** folder.

Each Jupyter Notebook utilizes Python scripts in the **Modules** folder. The **Activations** subfolder contains a modified implementation of the *softplus* nonlinear activation function. The **Loaders** subfolder contains a script for loading the scratch assay data, rescaling the variables, and removing outliers. The **Models** subfolder contains scripts for constructing multilayer perceptron (MLP) neural network models as well as a script for constructing BINNs (which are composed of a surrogate model for the dynamical system, u<sub>MLP</sub>, and the diffusion, D<sub>MLP</sub>, growth, G<sub>MLP</sub>, and delay, T<sub>MLP</sub>, networks). Note that the BINNs model class also includes the GLS, PDE, and constraint loss terms. The **Utils** subfolder contains utility functions for various tasks including importing relevant subpackages, wrapping the BINN model for easy training, saving, loading, etc., computing gradients, and forward solving PDEs. 

#### Citation

```
@article{lagergren2020biologicallyinformed,  
    title={Biologically-informed neural networks guide mechanistic modeling from sparse experimental data},  
    author={John H. Lagergren and John T. Nardini and Ruth E. Baker and Matthew J. Simpson and Kevin B. Flores},  
    journal={arXiv preprint arXiv:2005.13073},  
    year={2020}  
}
```

#### References

1. Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks:  Adeep learning framework for solving forward and inverse problems involvingnonlinear partial differential equations. *Journal of Computational Physics.* 2019;378:686 – 707.

2. Jin W, Shah ET, Penington CJ, McCue SW, Chopin LK, Simpson MJ.Reproducibility of scratch assays is affected by the initial degree of confluence:Experiments, modelling and model selection. *Journal of Theoretical Biology.* 2016;390:136 – 145.
