"""
Classes for describing the models in :py:mod:`mtenn.model`.

``mtenn`` makes heavy use of ``pydantic`` schema to ensure reproducibility and
self-documentation. Each model implemented in :py:mod:`mtenn.conversion_utils` must also
have a corresponding config class here.

These config classes act as a description of a model, which can then be instantiated
using the config's ``build`` method.
This ``build`` method must be implemented for any sub-class of the abstract
:py:class:`ModelConfigBase <mtenn.config.ModelConfigBase>`.
"""

from __future__ import annotations

import abc
from enum import Enum
from pydantic import BaseModel, Field, root_validator
import random
from typing import Callable, ClassVar
import mtenn.combination
import mtenn.readout
import mtenn.model
import numpy as np
import torch


class StringEnum(str, Enum):
    """
    Helper class with some convenience functions for our ``Type`` classes below.
    """

    @classmethod
    def get_values(cls) -> list[str]:
        """
        Get a list of all ``Enum`` values.
        """
        return [member.value for member in cls]

    @classmethod
    def reverse_lookup(cls, value):
        """
        Get the ``Enum`` entry corresponding to ``value``.

        Parameters
        ----------
        value : str
            String value of the ``Enum`` entry

        Returns
        -------
        cls
            ``Enum`` entry corresponding to ``value``
        """
        return cls(value)

    @classmethod
    def get_names(cls) -> list[str]:
        """
        Get a list of all ``Enum`` names.
        """
        return [member.name for member in cls]


class ModelType(StringEnum):
    """
    Enum for model types. See :py:mod:`mtenn.conversion_utils` for more details on
    the models.

    * GAT: Graph Attention Network (:py:class:`GAT <mtenn.conversion_utils.gat.GAT>`)

    * schnet: (:py:class:`SchNet <mtenn.conversion_utils.schnet.SchNet>`)

    * e3nn: E(3)-equivariant neural network
      (:py:class:`E3NN <mtenn.conversion_utils.e3nn.E3NN>`)

    * visnet: (:py:class:`ViSNet <mtenn.conversion_utils.visnet.ViSNet>`)

    * INVALID: Invalid model type to catch instantiation errors
    """

    GAT = "GAT"
    pyg_gat = "pyg_gat"
    dgl_gat = "dgl_gat"
    schnet = "schnet"
    e3nn = "e3nn"
    visnet = "visnet"
    INVALID = "INVALID"


class StrategyConfig(StringEnum):
    """
    Enum for possible ``mtenn`` Strategy classes. See :py:mod:`mtenn.strategy` for
    more details on each strategy.

    * delta: :py:class:`DeltaStrategy <mtenn.strategy.DeltaStrategy>`

    * concat: :py:class:`ConcatStrategy <mtenn.strategy.ConcatStrategy>`

    * complex: :py:class:`ComplexOnlyStrategy <mtenn.strategy.ComplexOnlyStrategy>`
    """

    # delta G strategy
    delta = "delta"
    # ML concatenation strategy
    concat = "concat"
    # Complex-only strategy
    complex = "complex"


class ReadoutConfig(StringEnum):
    """
    Enum for possible ``mtenn`` Readout classes. See :py:mod:`mtenn.readout` for
    more details on each readout option.

    * pic50: :py:class:`PIC50Readout <mtenn.readout.PIC50Readout>`

    * pki: :py:class:`PKiReadout <mtenn.readout.PKiReadout>`
    """

    pic50 = "pic50"
    pki = "pki"


class CombinationConfig(StringEnum):
    """
    Enum for possible ``mtenn`` Combination classes. See :py:mod:`mtenn.combination` for
    more details on each combination option.

    * mean: :py:class:`MeanCombination <mtenn.combination.MeanCombination>`

    * max: :py:class:`MaxCombination <mtenn.combination.MaxCombination>`
    """

    mean = "mean"
    max = "max"


class ModelConfigBase(BaseModel):
    """
    Abstract base class that model config classes will subclass. Any subclass needs
    to implement the ``_build`` method in order to be used.
    """

    model_type: ModelType = Field(ModelType.INVALID, const=True, allow_mutation=False)

    # Random seed optional for reproducibility
    rand_seed: int | None = Field(
        None, type=int, description="Random seed to set for Python, PyTorch, and NumPy."
    )

    # Model weights
    model_weights: dict | None = Field(None, type=dict, description="Model weights.")

    # Shared parameters for MTENN
    grouped: bool = Field(False, description="Model is a grouped (multi-pose) model.")
    strategy: StrategyConfig = Field(
        StrategyConfig.delta,
        description=(
            "Which ``Strategy`` to use for combining complex, protein, and ligand "
            "representations in the ``mtenn.Model``. "
            f"Options are [{', '.join(StrategyConfig.get_values())}]."
        ),
    )
    pred_readout: ReadoutConfig | None = Field(
        None,
        description=(
            "Which ``Readout`` to use for the model predictions. This corresponds "
            "to the individual pose predictions in the case of a ``GroupedModel``. "
            f"Options are [{', '.join(ReadoutConfig.get_values())}]."
        ),
    )
    combination: CombinationConfig | None = Field(
        None,
        description=(
            "Which ``Combination`` to use for combining predictions in a "
            "``GroupedModel``. "
            f"Options are [{', '.join(CombinationConfig.get_values())}]."
        ),
    )
    comb_readout: ReadoutConfig | None = Field(
        None,
        description=(
            "Which ``Readout`` to use for the combined model predictions. This is only "
            "relevant in the case of a ``GroupedModel``. "
            f"Options are [{', '.join(ReadoutConfig.get_values())}]."
        ),
    )

    # Parameters for MaxCombination
    max_comb_neg: bool = Field(
        True,
        description=(
            "Whether to take the min instead of max when combining pose predictions "
            "with ``MaxCombination``."
        ),
    )
    max_comb_scale: float = Field(
        1000,
        description=(
            "Scaling factor for values when taking the max/min when combining pose "
            "predictions with ``MaxCombination``. A value of 1 will approximate the "
            "Boltzmann mean, while a larger value will more accurately approximate the "
            "max/min operation."
        ),
    )

    # Parameters for PIC50Readout for pred_readout
    pred_substrate: float | None = Field(
        None,
        description=(
            "Substrate concentration to use when using the Cheng-Prusoff equation to "
            "convert :math:`\Delta G` -> :math:`\mathrm{IC_{50}}` in ``PIC50Readout`` "
            "for ``pred_readout``. Assumed to be in the same units as ``pred_km``."
        ),
    )
    pred_km: float | None = Field(
        None,
        description=(
            ":math:`\mathrm{K_m}` value to use when using the Cheng-Prusoff equation "
            "to convert :math:`\Delta G` -> :math:`\mathrm{IC_{50}}` in "
            "``PIC50Readout`` for ``pred_readout``. Assumed to be in the same units as "
            "``pred_substrate``."
        ),
    )

    # Parameters for PIC50Readout for comb_readout
    comb_substrate: float | None = Field(
        None,
        description=(
            "Substrate concentration to use when using the Cheng-Prusoff equation to "
            "convert :math:`\Delta G` -> :math:`\mathrm{IC_{50}}` in ``PIC50Readout`` "
            "for ``comb_readout``. Assumed to be in the same units as ``comb_km``."
        ),
    )
    comb_km: float | None = Field(
        None,
        description=(
            ":math:`\mathrm{K_m}` value to use when using the Cheng-Prusoff equation "
            "to convert :math:`\Delta G` -> :math:`\mathrm{IC_{50}}` in "
            "``PIC50Readout`` for ``comb_readout``. Assumed to be in the same units as "
            "``comb_substrate``."
        ),
    )

    class Config:
        validate_assignment = True

    def build(self) -> mtenn.model.Model:
        """
        Exposed function that first parses all the ``mtenn``-related args, and then
        calls the ``_build`` method to construct the
        :py:class:`Model <mtenn.model.Model>` object.

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        # First set random seeds if applicable
        if self.rand_seed is not None:
            random.seed(self.rand_seed)
            torch.manual_seed(self.rand_seed)
            np.random.seed(self.rand_seed)

        # First handle the MTENN classes
        match self.combination:
            case CombinationConfig.mean:
                mtenn_combination = mtenn.combination.MeanCombination()
            case CombinationConfig.max:
                mtenn_combination = mtenn.combination.MaxCombination(
                    negate_preds=self.max_comb_neg, pred_scale=self.max_comb_scale
                )
            case None:
                mtenn_combination = None

        match self.pred_readout:
            case ReadoutConfig.pic50:
                mtenn_pred_readout = mtenn.readout.PIC50Readout(
                    substrate=self.pred_substrate, Km=self.pred_km
                )
            case ReadoutConfig.pki:
                mtenn_pred_readout = mtenn.readout.PKiReadout()
            case None:
                mtenn_pred_readout = None

        match self.comb_readout:
            case ReadoutConfig.pic50:
                mtenn_comb_readout = mtenn.readout.PIC50Readout(
                    substrate=self.comb_substrate, Km=self.comb_km
                )
            case ReadoutConfig.pki:
                mtenn_comb_readout = mtenn.readout.PKiReadout()
            case None:
                mtenn_comb_readout = None

        mtenn_params = {
            "combination": mtenn_combination,
            "pred_readout": mtenn_pred_readout,
            "comb_readout": mtenn_comb_readout,
        }

        # Build the actual Model
        model = self._build(mtenn_params)

        # Set model weights
        if self.model_weights:
            model.load_state_dict(self.model_weights)

        return model

    @abc.abstractmethod
    def _build(self, mtenn_params={}) -> mtenn.model.Model:
        """
        Method that actually builds the :py:class:`Model <mtenn.model.Model>` object.
        Must be implemented for any subclass.

        :meta public:

        Parameters
        ----------
        mtenn_params : dict, optional
            Dictionary that stores the ``Readout`` objects for the individual
            predictions and for the combined prediction, and the ``Combination`` object
            in the case of a multi-pose model. These are all constructed the same for all
            ``Model`` types, so we can just handle them in the base class. Keys in the
            dict will be:

            * "combination": :py:mod:`Combination <mtenn.combination>`

            * "pred_readout": :py:mod:`Readout <mtenn.readout>` for individual
              pose predictions

            * "comb_readout": :py:mod:`Readout <mtenn.readout>` for combined
              prediction (in the case of a multi-pose model)

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        ...

    def update(self, config_updates={}) -> ModelConfigBase:
        """
        Create a new config object with field values replaced by any given in
        ``config_updates``. Note that this is NOT an in-place operation, and will return
        a new config object, leaving the original un-modified.

        This function just wraps around the ``_update`` function, which can be overloaded
        in any subclasses. A default ``_update`` implementation that should work for
        most cases is provided.

        Parameters
        ----------
        config_updates : dict
            Dictionary mapping from field names to new values

        Returns
        -------
        cls
            Returns an object that is the same type as the calling object
        """
        return self._update(config_updates)

    def _update(self, config_updates={}) -> ModelConfigBase:
        """
        Default version of this function. Just update original config with new options,
        and generate new object. Designed to be overloaded if there are specific things
        that a class needs to handle (see
        :py:class:`GATModelConfig <mtenn.config.GATModelConfig>` as an example).

        :meta public:

        Parameters
        ----------
        config_updates : dict
            Dictionary mapping from field names to new values

        Returns
        -------
        cls
            Returns an object that is the same type as the calling object
        """

        orig_config = self.dict()

        # Get new config by overwriting old stuff with any new stuff
        new_config = orig_config | config_updates

        return type(self)(**new_config)

    @staticmethod
    def _check_grouped(values):
        """
        Makes sure that a Combination method is passed if using a GroupedModel. Only
        needs to be called for structure-based models.
        """
        if values["grouped"] and (not values["combination"]):
            raise ValueError("combination must be specified for a GroupedModel.")


class GATModelConfig(ModelConfigBase):
    """
    Class for constructing a GAT ML model. Default values here are based on the values
    in DGL-LifeSci.
    """

    model_type: ModelType = Field(ModelType.GAT, const=True)

    in_channels: int = Field(
        -1,
        description=(
            "Input size. Can be left as -1 (default) to interpret based on "
            "first forward call."
        ),
    )
    hidden_channels: int = Field(32, description="Hidden embedding size.")
    num_layers: int = Field(2, description="Number of GAT layers.")
    v2: bool = Field(False, description="Use GATv2Conv layer instead of GATConv.")
    dropout: float = Field(0, description="Dropout probability.")
    heads: int = Field(4, description="Number of attention heads for each GAT layer.")
    negative_slope: float = Field(
        0.2, description="LeakyReLU angle of the negative slope."
    )

    def _build(self, mtenn_params={}):
        """
        Build an ``mtenn`` GAT ``Model`` from this config.

        :meta public:

        Parameters
        ----------
        mtenn_params : dict, optional
            Dictionary that stores the ``Readout`` objects for the individual
            predictions and for the combined prediction, and the ``Combination`` object
            in the case of a multi-pose model. These are all constructed the same for all
            ``Model`` types, so we can just handle them in the base class. Keys in the
            dict will be:

            * "combination": :py:mod:`Combination <mtenn.combination>`

            * "pred_readout": :py:mod:`Readout <mtenn.readout>` for individual
              pose predictions

            * "comb_readout": :py:mod:`Readout <mtenn.readout>` for combined
              prediction (in the case of a multi-pose model)

            although the combination-related entries will be ignore because this is a
            ligand-only model.

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        from mtenn.conversion_utils.gat import GAT

        model = GAT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            v2=self.v2,
            dropout=self.dropout,
            heads=self.heads,
            negative_slope=self.negative_slope,
        )

        pred_readout = mtenn_params.get("pred_readout", None)
        return GAT.get_model(model=model, pred_readout=pred_readout, fix_device=True)


class PyGGATModelConfig(GATModelConfig):
    model_type: ModelType = Field(ModelType.pyg_gat, const=True)

    def _build(self, mtenn_params={}):
        """
        Build an ``mtenn`` PyGGAT ``Model`` from this config.

        :meta public:

        Parameters
        ----------
        mtenn_params : dict, optional
            Dictionary that stores the ``Readout`` objects for the individual
            predictions and for the combined prediction, and the ``Combination`` object
            in the case of a multi-pose model. These are all constructed the same for all
            ``Model`` types, so we can just handle them in the base class. Keys in the
            dict will be:

            * "combination": :py:mod:`Combination <mtenn.combination>`

            * "pred_readout": :py:mod:`Readout <mtenn.readout>` for individual
              pose predictions

            * "comb_readout": :py:mod:`Readout <mtenn.readout>` for combined
              prediction (in the case of a multi-pose model)

            although the combination-related entries will be ignore because this is a
            ligand-only model.

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        from mtenn.conversion_utils.pyg_gat import PyGGAT

        model = PyGGAT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            v2=self.v2,
            dropout=self.dropout,
            heads=self.heads,
            negative_slope=self.negative_slope,
        )

        pred_readout = mtenn_params.get("pred_readout", None)
        return PyGGAT.get_model(model=model, pred_readout=pred_readout, fix_device=True)


class DGLGATModelConfig(ModelConfigBase):
    """
    Class for constructing a graph attention ML model. Note that there are two methods
    for defining the size of the model:

    * If single values are passed for all parameters, the value of ``num_layers`` will
      be used as the size of the model, and each layer will have the parameters given

    * If a list of values is passed for any parameters, all parameters must be lists of
      the same size, or single values. For parameters that are single values, that same
      value will be used for each layer. For parameters that are lists, those lists will
      be used

    Parameters passed as strings are assumed to be comma-separated lists, and will first
    be cast to lists of the appropriate type, and then processed as described above.

    If lists of multiple different (non-1) sizes are found, an error will be raised.

    Default values here are the default values given in DGL-LifeSci.
    """

    # Import as private, mainly so Sphinx doesn't autodoc it
    from dgllife.utils import CanonicalAtomFeaturizer as _CanonicalAtomFeaturizer

    # Dict of model params that can be passed as a list, and the type that each will be
    #  cast to
    LIST_PARAMS: ClassVar[dict] = {
        "hidden_feats": int,
        "num_heads": int,
        "feat_drops": float,
        "attn_drops": float,
        "alphas": float,
        "residuals": bool,
        "agg_modes": str,
        "activations": None,
        "biases": bool,
    }  #: :meta private:

    model_type: ModelType = Field(ModelType.dgl_gat, const=True)

    in_feats: int = Field(
        _CanonicalAtomFeaturizer().feat_size(),
        description=(
            "Input node feature size. Defaults to size of the "
            "``CanonicalAtomFeaturizer``."
        ),
    )
    num_layers: int = Field(
        2,
        description=(
            "Number of GAT layers. Ignored if a list of values is passed for any "
            "other argument."
        ),
    )
    hidden_feats: str | int | list[int] = Field(
        32,
        description=(
            "Output size of each GAT layer. If an ``int`` is passed, the value for "
            "``num_layers`` will be used to determine the size of the model. If a list "
            "of ``int`` s is passed, the size of the model will be inferred from the "
            "length of the list."
        ),
    )
    num_heads: str | int | list[int] = Field(
        4,
        description=(
            "Number of attention heads for each GAT layer. Passing an ``int`` or list "
            "of ``int`` s functions similarly as for ``hidden_feats``."
        ),
    )
    feat_drops: str | float | list[float] = Field(
        0,
        description=(
            "Dropout of input features for each GAT layer. Passing a ``float`` or "
            "list of ``float`` s functions similarly as for ``hidden_feats``."
        ),
    )
    attn_drops: str | float | list[float] = Field(
        0,
        description=(
            "Dropout of attention values for each GAT layer. Passing a ``float`` or "
            "list of ``float`` s functions similarly as for ``hidden_feats``."
        ),
    )
    alphas: str | float | list[float] = Field(
        0.2,
        description=(
            "Hyperparameter for ``LeakyReLU`` gate for each GAT layer. Passing a "
            "``float`` or list of ``float`` s functions similarly as for "
            "``hidden_feats``."
        ),
    )
    residuals: str | bool | list[bool] = Field(
        True,
        description=(
            "Whether to use residual connection for each GAT layer. Passing a ``bool`` "
            "or list of ``bool`` s functions similarly as for ``hidden_feats``."
        ),
    )
    agg_modes: str | list[str] = Field(
        "flatten",
        description=(
            "Which aggregation mode [flatten, mean] to use for each GAT layer. "
            "Passing a ``str`` or list of ``str`` s functions similarly as for "
            "``hidden_feats``."
        ),
    )
    activations: Callable | list[Callable] | list[None] | None = Field(
        None,
        description=(
            "Activation function for each GAT layer. Passing a function or "
            "list of functions functions similarly as for ``hidden_feats``."
        ),
    )
    biases: str | bool | list[bool] = Field(
        True,
        description=(
            "Whether to use bias for each GAT layer. Passing a ``bool`` or "
            "list of ``bool`` s functions similarly as for ``hidden_feats``."
        ),
    )
    allow_zero_in_degree: bool = Field(
        False, description="Allow zero in degree nodes for all graph layers."
    )

    # Internal tracker for if the parameters were originally built from lists or using
    #  num_layers
    _from_num_layers = False

    @root_validator(pre=False)
    def massage_into_lists(cls, values) -> DGLGATModelConfig:
        """
        Validator to handle unifying all the values into the proper list forms based on
        the rules described in the class docstring.
        """
        # First convert string lists to actual lists
        for param, param_type in cls.LIST_PARAMS.items():
            param_val = values[param]
            if isinstance(param_val, str):
                try:
                    param_val = list(map(param_type, param_val.split(",")))
                except ValueError:
                    raise ValueError(
                        f"Unable to parse value {param_val} for parameter {param}. "
                        f"Expected type of {param_type}."
                    )
                values[param] = param_val

        # Get sizes of all lists
        list_lens = {}
        for p in cls.LIST_PARAMS:
            param_val = values[p]
            if not isinstance(param_val, list):
                # Shouldn't be possible at this point but just in case
                param_val = [param_val]
                values[p] = param_val
            list_lens[p] = len(param_val)

        # Check that there's only one length present
        list_lens_set = set(list_lens.values())
        # This could be 0 if lists of length 1 were passed, which is valid
        if len(list_lens_set - {1}) > 1:
            raise ValueError(
                "All passed parameter lists must be the same value. "
                f"Instead got list lengths of: {list_lens}"
            )
        elif list_lens_set == {1}:
            # If all lists have only one value, we defer to the value passed to
            #  num_layers, as described in the class docstring
            num_layers = values["num_layers"]
            values["_from_num_layers"] = True
        else:
            num_layers = max(list_lens_set)
            values["_from_num_layers"] = False

        values["num_layers"] = num_layers
        # If we just want a model with one layer, can return early since we've already
        #  converted everything into lists
        if num_layers == 1:
            return values

        # Adjust any length 1 list to be the right length
        for p, list_len in list_lens.items():
            if list_len == 1:
                values[p] = values[p] * num_layers

        return values

    def _build(self, mtenn_params={}):
        """
        Build an ``mtenn`` GAT ``Model`` from this config.

        :meta public:

        Parameters
        ----------
        mtenn_params : dict, optional
            Dictionary that stores the ``Readout`` objects for the individual
            predictions and for the combined prediction, and the ``Combination`` object
            in the case of a multi-pose model. These are all constructed the same for all
            ``Model`` types, so we can just handle them in the base class. Keys in the
            dict will be:

            * "combination": :py:mod:`Combination <mtenn.combination>`

            * "pred_readout": :py:mod:`Readout <mtenn.readout>` for individual
              pose predictions

            * "comb_readout": :py:mod:`Readout <mtenn.readout>` for combined
              prediction (in the case of a multi-pose model)

            although the combination-related entries will be ignore because this is a
            ligand-only model.

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        from mtenn.conversion_utils.dgl_gat import DGLGAT

        model = DGLGAT(
            in_feats=self.in_feats,
            hidden_feats=self.hidden_feats,
            num_heads=self.num_heads,
            feat_drops=self.feat_drops,
            attn_drops=self.attn_drops,
            alphas=self.alphas,
            residuals=self.residuals,
            agg_modes=self.agg_modes,
            activations=self.activations,
            biases=self.biases,
            allow_zero_in_degree=self.allow_zero_in_degree,
        )

        pred_readout = mtenn_params.get("pred_readout", None)
        return DGLGAT.get_model(model=model, pred_readout=pred_readout, fix_device=True)

    def _update(self, config_updates={}) -> DGLGATModelConfig:
        """
        GAT-specific implementation of updating logic. Need to handle stuff specially
        to make sure that the original method of specifying parameters (either from a
        passed value of ``num_layers`` or inferred from each parameter being a list) is
        maintained.

        :meta public:

        Parameters
        ----------
        config_updates : dict
            Dictionary mapping from field names to new values

        Returns
        -------
        DGLGATModelConfig
            New ``DGLGATModelConfig`` object
        """
        orig_config = self.dict()
        if self._from_num_layers or ("num_layers" in config_updates):
            # If originally generated from num_layers, want to pull out the first entry
            #  in each list param so it can be re-broadcast with (potentially) new
            #  num_layers
            for param_name in DGLGATModelConfig.LIST_PARAMS.keys():
                orig_config[param_name] = orig_config[param_name][0]

        # Get new config by overwriting old stuff with any new stuff
        new_config = orig_config | config_updates

        # A bit hacky, maybe try and change?
        if isinstance(new_config["activations"], list) and (
            new_config["activations"][0] is None
        ):
            new_config["activations"] = None

        return DGLGATModelConfig(**new_config)


class SchNetModelConfig(ModelConfigBase):
    """
    Class for constructing a SchNet ML model. Default values here are the default values
    given in PyG.
    """

    model_type: ModelType = Field(ModelType.schnet, const=True)

    hidden_channels: int = Field(128, description="Hidden embedding size.")
    num_filters: int = Field(
        128, description="Number of filters to use in the cfconv layers."
    )
    num_interactions: int = Field(6, description="Number of interaction blocks.")
    num_gaussians: int = Field(
        50, description="Number of gaussians to use in the interaction blocks."
    )
    interaction_graph: Callable | None = Field(
        None,
        description=(
            "Function to compute the pairwise interaction graph and "
            "interatomic distances."
        ),
    )
    cutoff: float = Field(
        10, description="Cutoff distance for interatomic interactions."
    )
    max_num_neighbors: int = Field(
        32, description="Maximum number of neighbors to collect for each node."
    )
    readout: str = Field(
        "add", description="Which global aggregation to use [add, mean]."
    )
    dipole: bool = Field(
        False,
        description=(
            "Whether to use the magnitude of the dipole moment to make the "
            "final prediction."
        ),
    )
    mean: float | None = Field(
        None,
        description=(
            "Mean of property to predict, to be added to the model prediction before "
            "returning. This value is only used if dipole is False and a value is also "
            "passed for std."
        ),
    )
    std: float | None = Field(
        None,
        description=(
            "Standard deviation of property to predict, used to scale the model "
            "prediction before returning. This value is only used if dipole is False "
            "and a value is also passed for mean."
        ),
    )
    atomref: list[float] | None = Field(
        None,
        description=(
            "Reference values for single-atom properties. Should have length of 100 to "
            "match with PyG."
        ),
    )

    @root_validator(pre=False)
    def validate(cls, values):
        # Make sure the grouped stuff is properly assigned
        ModelConfigBase._check_grouped(values)

        # Make sure atomref length is correct (this is required by PyG)
        atomref = values["atomref"]
        if (atomref is not None) and (len(atomref) != 100):
            raise ValueError(f"atomref must be length 100 (got {len(atomref)})")

        return values

    def _build(self, mtenn_params={}):
        """
        Build an ``mtenn`` SchNet ``Model`` from this config.

        :meta public:

        Parameters
        ----------
        mtenn_params : dict, optional
            Dictionary that stores the ``Readout`` objects for the individual
            predictions and for the combined prediction, and the ``Combination`` object
            in the case of a multi-pose model. These are all constructed the same for all
            ``Model`` types, so we can just handle them in the base class. Keys in the
            dict will be:

            * "combination": :py:mod:`Combination <mtenn.combination>`

            * "pred_readout": :py:mod:`Readout <mtenn.readout>` for individual
              pose predictions

            * "comb_readout": :py:mod:`Readout <mtenn.readout>` for combined
              prediction (in the case of a multi-pose model)

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        from mtenn.conversion_utils.schnet import SchNet

        # Create an MTENN SchNet model from PyG SchNet model
        model = SchNet(
            hidden_channels=self.hidden_channels,
            num_filters=self.num_filters,
            num_interactions=self.num_interactions,
            num_gaussians=self.num_gaussians,
            interaction_graph=self.interaction_graph,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
            readout=self.readout,
            dipole=self.dipole,
            mean=self.mean,
            std=self.std,
            atomref=self.atomref,
        )

        combination = mtenn_params.get("combination", None)
        pred_readout = mtenn_params.get("pred_readout", None)
        comb_readout = mtenn_params.get("comb_readout", None)

        return SchNet.get_model(
            model=model,
            grouped=self.grouped,
            fix_device=True,
            strategy=self.strategy,
            combination=combination,
            pred_readout=pred_readout,
            comb_readout=comb_readout,
        )


class E3NNModelConfig(ModelConfigBase):
    """
    Class for constructing an e3nn ML model.
    """

    model_type: ModelType = Field(ModelType.e3nn, const=True)

    num_atom_types: int = Field(
        100,
        description=(
            "Number of different atom types. In general, this will just be the "
            "max atomic number of all input atoms."
        ),
    )
    irreps_hidden: dict[str, int] | str = Field(
        {"0": 10, "1": 3, "2": 2, "3": 1},
        description=(
            "``Irreps`` for the hidden layers of the network. "
            "This can either take the form of an ``Irreps`` string, or a dict mapping "
            ":math:`\\mathcal{l}` levels (parity optional) to the number of ``Irreps`` "
            "of that level. "
            "If parity is not passed for a given level, both parities will be used. If "
            "you only want one parity for a given level, make sure you specify it. "
            "A dict can also be specified as a string, in the format of a comma "
            "separated list of ``<irreps_l>:<num_irreps>``."
        ),
    )
    lig: bool = Field(
        False, description="Include ligand labels as a node attribute information."
    )
    irreps_edge_attr: int = Field(
        3,
        description=(
            "Which level of spherical harmonics to use for encoding edge attributes "
            "internally."
        ),
    )
    num_layers: int = Field(3, description="Number of network layers.")
    neighbor_dist: float = Field(
        10, description="Cutoff distance for including atoms as neighbors."
    )
    num_basis: int = Field(
        10, description="Number of bases on which the edge length are projected."
    )
    num_radial_layers: int = Field(1, description="Number of radial layers.")
    num_radial_neurons: int = Field(
        128, description="Number of neurons in each radial layer."
    )
    num_neighbors: float = Field(25, description="Typical number of neighbor nodes.")
    num_nodes: float = Field(4700, description="Typical number of nodes in a graph.")

    @root_validator(pre=False)
    def massage_irreps(cls, values):
        """
        Check that the value given for ``irreps_hidden`` can be converted into an Irreps
        representation, and do so.
        """
        from e3nn import o3

        # First just check that the grouped stuff is properly assigned
        ModelConfigBase._check_grouped(values)

        # Now deal with irreps
        irreps = values["irreps_hidden"]
        # First see if this string should be converted into a dict
        if isinstance(irreps, str):
            if ":" in irreps:
                orig_irreps = irreps
                irreps = [i.split(":") for i in irreps.split(",")]
                try:
                    irreps = {
                        irreps_l: int(num_irreps) for irreps_l, num_irreps in irreps
                    }
                except ValueError:
                    raise ValueError(
                        f"Unable to parse irreps dict string: {orig_irreps}"
                    )
            else:
                # If not, try and convert directly to Irreps
                try:
                    _ = o3.Irreps(irreps)
                except ValueError:
                    raise ValueError(f"Invalid irreps string: {irreps}")

                # If already in a good string, can just return
                return values

        # If we got a dict, need to massage that into an Irreps string
        # First make a copy of the input dict in case of errors
        orig_irreps = irreps.copy()
        # Find L levels that got an unspecified parity
        unspecified_l = [k for k in irreps.keys() if ("o" not in k) and ("e" not in k)]
        for irreps_l in unspecified_l:
            num_irreps = irreps.pop(irreps_l)
            irreps[f"{irreps_l}o"] = num_irreps
            irreps[f"{irreps_l}e"] = num_irreps

        # Combine Irreps into str
        irreps = "+".join(
            [
                f"{num_irreps}x{irrep}"
                for irrep, num_irreps in irreps.items()
                if num_irreps > 0
            ]
        )

        # Make sure this Irreps string is valid
        try:
            _ = o3.Irreps(irreps)
        except ValueError:
            raise ValueError(f"Couldn't parse irreps dict: {orig_irreps}")

        values["irreps_hidden"] = irreps
        return values

    def _build(self, mtenn_params={}):
        """
        Build an ``mtenn`` e3nn ``Model`` from this config.

        :meta public:

        Parameters
        ----------
        mtenn_params : dict, optional
            Dictionary that stores the ``Readout`` objects for the individual
            predictions and for the combined prediction, and the ``Combination`` object
            in the case of a multi-pose model. These are all constructed the same for all
            ``Model`` types, so we can just handle them in the base class. Keys in the
            dict will be:

            * "combination": :py:mod:`Combination <mtenn.combination>`

            * "pred_readout": :py:mod:`Readout <mtenn.readout>` for individual
              pose predictions

            * "comb_readout": :py:mod:`Readout <mtenn.readout>` for combined
              prediction (in the case of a multi-pose model)

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        from e3nn.o3 import Irreps
        from mtenn.conversion_utils.e3nn import E3NN

        model = E3NN(
            irreps_in=f"{self.num_atom_types}x0e",
            irreps_hidden=self.irreps_hidden,
            irreps_out="1x0e",
            irreps_node_attr="1x0e" if self.lig else None,
            irreps_edge_attr=Irreps.spherical_harmonics(self.irreps_edge_attr),
            layers=self.num_layers,
            max_radius=self.neighbor_dist,
            number_of_basis=self.num_basis,
            radial_layers=self.num_radial_layers,
            radial_neurons=self.num_radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            reduce_output=True,
        )

        combination = mtenn_params.get("combination", None)
        pred_readout = mtenn_params.get("pred_readout", None)
        comb_readout = mtenn_params.get("comb_readout", None)

        return E3NN.get_model(
            model=model,
            grouped=self.grouped,
            fix_device=True,
            strategy=self.strategy,
            combination=combination,
            pred_readout=pred_readout,
            comb_readout=comb_readout,
        )


class ViSNetModelConfig(ModelConfigBase):
    """
    Class for constructing a VisNet ML model. Default values here are the default values
    given in PyG.
    """

    model_type: ModelType = Field(ModelType.visnet, const=True)
    lmax: int = Field(1, description="The maximum degree of the spherical harmonics.")
    vecnorm_type: str | None = Field(
        None, description="The type of normalization to apply to the vectors."
    )
    trainable_vecnorm: bool = Field(
        False, description="Whether the normalization weights are trainable."
    )
    num_heads: int = Field(8, description="The number of attention heads.")
    num_layers: int = Field(6, description="The number of layers in the network.")
    hidden_channels: int = Field(
        128, description="The number of hidden channels in the node embeddings."
    )
    num_rbf: int = Field(32, description="The number of radial basis functions.")
    trainable_rbf: bool = Field(
        False, description="Whether the radial basis function parameters are trainable."
    )
    max_z: int = Field(100, description="The maximum atomic numbers.")
    cutoff: float = Field(5.0, description="The cutoff distance.")
    max_num_neighbors: int = Field(
        32, description="The maximum number of neighbors considered for each atom."
    )
    vertex: bool = Field(False, description="Whether to use vertex geometric features.")
    atomref: list[float] | None = Field(
        None,
        description=(
            "Reference values for single-atom properties. Should have length ``max_z``."
        ),
    )
    reduce_op: str = Field(
        "sum",
        description=(
            "The type of reduction operation to apply. Options are [sum, mean]."
        ),
    )
    mean: float = Field(0.0, description="The mean of the output distribution.")
    std: float = Field(
        1.0, description="The standard deviation of the output distribution."
    )
    derivative: bool = Field(
        False,
        description=(
            "Whether to compute the derivative of the output with respect to the "
            "positions."
        ),
    )

    @root_validator(pre=False)
    def validate(cls, values):
        """
        Check that ``atomref`` and ``max_z`` agree.
        """
        # Make sure the grouped stuff is properly assigned
        ModelConfigBase._check_grouped(values)

        # Make sure atomref length is correct (this is required by PyG)
        atomref = values["atomref"]
        if (atomref is not None) and (len(atomref) != values["max_z"]):
            raise ValueError(
                f"atomref length must match max_z. (Expected {values['max_z']}, got {len(atomref)})"
            )

        return values

    def _build(self, mtenn_params={}):
        """
        Build an ``mtenn`` ViSNet ``Model`` from this config.

        :meta public:

        Parameters
        ----------
        mtenn_params : dict, optional
            Dictionary that stores the ``Readout`` objects for the individual
            predictions and for the combined prediction, and the ``Combination`` object
            in the case of a multi-pose model. These are all constructed the same for all
            ``Model`` types, so we can just handle them in the base class. Keys in the
            dict will be:

            * "combination": :py:mod:`Combination <mtenn.combination>`

            * "pred_readout": :py:mod:`Readout <mtenn.readout>` for individual
              pose predictions

            * "comb_readout": :py:mod:`Readout <mtenn.readout>` for combined
              prediction (in the case of a multi-pose model)

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        # Create an MTENN ViSNet model from PyG ViSNet model
        from mtenn.conversion_utils.visnet import ViSNet

        model = ViSNet(
            lmax=self.lmax,
            vecnorm_type=self.vecnorm_type,
            trainable_vecnorm=self.trainable_vecnorm,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            hidden_channels=self.hidden_channels,
            num_rbf=self.num_rbf,
            trainable_rbf=self.trainable_rbf,
            max_z=self.max_z,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
            vertex=self.vertex,
            reduce_op=self.reduce_op,
            mean=self.mean,
            std=self.std,
            derivative=self.derivative,
            atomref=self.atomref,
        )
        combination = mtenn_params.get("combination", None)
        pred_readout = mtenn_params.get("pred_readout", None)
        comb_readout = mtenn_params.get("comb_readout", None)

        return ViSNet.get_model(
            model=model,
            grouped=self.grouped,
            fix_device=True,
            strategy=self.strategy,
            combination=combination,
            pred_readout=pred_readout,
            comb_readout=comb_readout,
        )
