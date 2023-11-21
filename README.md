# Gravity-(V)GAE Pytorch Geometric

This repository contains a Pytorch Geometric implementation of the [Gravity-Inspired Graph Autoencoders for Directed Link Prediction](https://doi.org/10.1145/3357384.3358023) paper.

With resepect to the [original implementation](https://github.com/deezer/gravity_graph_autoencoders), it has the following limitations:

- Scripts are provided only for the Cora dataset;
- Only the Gravity-GAE, Gravity-VGAE, SourceTarget-GAE and SourceTarget-VGAE are implemented.

And additions:

- Early stopping;
- Trainable lambda;
- New training strategy for the biased task, which sometimes drastically improves performance

See below for details on each of these additions.

## Early Stopping

The general setting contains a validation set that we use to perform early stopping w.r.t. a validation loss obtained by summing the AUC and AP scores on the validation set.

## Trainable Lambda

Instead of treating lambda as an hyperparameter, we enable training for it.

## New Training Strategy

The "*_Rev.py" scripts contained in the `gravity_gae/mains/Biased` folder have a train set which is different from that of the original implementation. It consists in all positive edges (as defined in the biased setting) and all their reverses as negatives. This produces a training set which is statistically more similar to the test set and thus improves performance on many models.

## Further Observations

Sometimes the `recon_loss` is more numerically stable than the `BCELossWithLogits`, and thus we preferred it.

## How to Run

Please use the provided `environment.yml` file to instantiate a conda environment `gravity_gae_env` compatible with this code. Then open the desired script in `gravity_gae/mains/` and run it e.g. in VSCode.

## References

Salha, G., Limnios, S., Hennequin, R., Tran, V., & Vazirgiannis, M. (2019). Gravity-Inspired Graph Autoencoders for Directed Link Prediction. In ACM International Conference on Information and Knowledge Management (CIKM).