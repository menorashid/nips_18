def get(name,args):
    import importlib
    if args is not None:
    	return importlib.import_module("dataset.%s" % name).Dataset(**args)
    else:
    	return importlib.import_module("dataset.%s" % name).Dataset()
