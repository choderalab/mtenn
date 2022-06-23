"""
Representation and strategy for SchNet model.
"""

def get_reprentation(model):
    """
    Input model, remove last layer.

    Parameters
    ----------

    Returns
    -------
    """

def get_energy_func(model):
    """
    Return last layer of the model.

    Parameters
    ----------

    Returns
    -------
    """

def get_delta_strategy(model):
    return(DeltaStrategy(get_energy_func(model)))

def get_model(model, strategy: str='delta'):
    if strategy == 'delta':
        strategy = get_delta_strategy
    elif strategy == 'concat':
        strategy = get_concat_strategy
    return(Model(get_reprentation(model), strategy(model)))