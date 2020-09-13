class Games:
    DRTS = "deep_rts"
    SC2 = "starcraft_ii"


class RTSGeneralization:
    METHOD_SINGLE = "single_environment"
    METHOD_MULTIPLE = "multiple_environment"

    STATE_MAP = "map"
    STATE_NON_SPATIAL = "non_spatial_only"
    STATE_BOTH = "map_and_non_spatial"
    STATE_MAXIMUM_X = 64
    STATE_MAXIMUM_Y = 64
    STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS = 20 
