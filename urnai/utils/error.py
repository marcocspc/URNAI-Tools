class ActionError(Exception):
    pass

class DeprecatedCodeException(Exception):
    pass

class CommandsNotUniqueError(Exception):
    pass

class EnvironmentNotSupportedError(Exception):
    pass

class MapNotFoundError(Exception):
    pass

class WadNotFoundError(Exception):
    pass

class IncoherentBuildModelError(Exception):
    pass

class UnsupportedBuildModelLayerTypeError(Exception):
    pass

class UnsupportedVizDoomRes(Exception):
    pass

class ClassNotFoundError(Exception):
    pass
