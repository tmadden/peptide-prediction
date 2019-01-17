def do_FMLN_encoding(peplist, m=8, n=3):
    """
    First m last n. e.g. for FELT encoding, the default, m=8, n=3

    :param peplist: the list of peptides to encode

    :param m: use the first m residues

    :param n: concatenated with the last n residues

    :returns: encoded peptide list
    """
    
    
    # e = []
    # for p in peplist:
    #     fmln = p[0:m] + p[-n:]
    #     e.append(fmln)

    # return e
    
    return [p[0:m] + p[-n:] for p in peplist]