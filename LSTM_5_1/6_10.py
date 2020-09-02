import numpy as np

Loc = [93]
    Sgroup = [i for i in range(13)]
    Pgroup = [i for i in range(13,93)]
    groups = {
        'S' : Sgroup,
        'P' : Pgroup,
        'SL' : Sgroup + Loc,
        'PL' : Pgroup + Loc,
        'SP' : Sgroup + Pgroup,
        'SPL' : Sgroup + Pgroup + Loc,
    }
    assert group in groups
    useGroup = groups['SPL']
    if forSOD == True:
        useGroup = [i for i in range(49)] #0-48
    X = X[:,:,useGroup]