def pdselect(dataframe, **conditions):
    '''Select rows from a dataframe according to conditions

    Example::

        pdselect(data, a=2, b__lt=3)

    will select all rows where 'a' is 2 and 'b' is less than 3
    '''
    import pandas as pd
    import numpy as np
    if type(dataframe) == pd.Series:
        dataframe = pd.DataFrame({'value': dataframe})
    for cond,value in conditions.items():
        if cond in dataframe.columns:
            cond = (dataframe[cond] == value)
        elif cond.endswith('__neq') or cond.endswith('__not_eq'):
            if cond.endswith('__neq'):
                cond = cond[:-len('__neq')]
            elif cond.endswith('__not_eq'):
                cond = cond[:-len('__not_eq')]
            cond = dataframe[cond] != value
        elif cond.endswith('__gt'):
            cond = cond[:-len('__gt')]
            cond = dataframe[cond] > value
        elif cond.endswith('__ge'):
            cond = cond[:-len('__ge')]
            cond = dataframe[cond] >= value
        elif cond.endswith('__gte'):
            cond = cond[:-len('__gte')]
            cond = dataframe[cond] >= value
        elif cond.endswith('__lt'):
            cond = cond[:-len('__lt')]
            cond = dataframe[cond] < value
        elif cond.endswith('__le'):
            cond = cond[:-len('__le')]
            cond = dataframe[cond] <= value
        elif cond.endswith('__lte'):
            cond = cond[:-len('__lte')]
            cond = dataframe[cond] <= value
        elif cond.endswith('__in'):
            cond = cond[:-len('__in')]
            cond = np.in1d(dataframe[cond], value)
        elif cond.endswith('__not_in'):
            cond = cond[:-len('__not_in')]
            cond = np.in1d(dataframe[cond], value, invert=True)
        else:
            raise ValueError("Cannot process condition '{}'".format(cond))
        dataframe = dataframe[cond]
    return dataframe
