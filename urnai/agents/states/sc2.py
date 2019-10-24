import math
import numpy as np
from .abstate import StateBuilder
from pysc2.lib import actions, features, units
from agents.actions.sc2 import * 

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_SCV = 45
_NEUTRAL_MINERAL_FIELD = 341

class Simple64State(StateBuilder):

    def __init__(self):
        #self._state_size = 22
        self._state_size = 8214

    def build_state(self, obs):
        if obs.game_loop[0] == 0:
            command_center = get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

        new_state = []
        new_state.append(obs.player.minerals)
        new_state.append(obs.player.vespene)
        new_state.append(obs.player.food_cap)
        new_state.append(obs.player.food_used)
        new_state.append(obs.player.food_army)
        new_state.append(obs.player.food_workers)
        new_state.append(obs.player.food_cap - obs.player.food_used)
        new_state.append(obs.player.army_count)
        new_state.append(obs.player.idle_worker_count)
        new_state.append(get_units_amount(obs, units.Terran.CommandCenter)+
                        get_units_amount(obs, units.Terran.OrbitalCommand)+
                        get_units_amount(obs, units.Terran.PlanetaryFortress))
        new_state.append(get_units_amount(obs, units.Terran.SupplyDepot))
        new_state.append(get_units_amount(obs, units.Terran.Refinery))
        new_state.append(get_units_amount(obs, units.Terran.EngineeringBay))
        new_state.append(get_units_amount(obs, units.Terran.Armory))
        new_state.append(get_units_amount(obs, units.Terran.MissileTurret))
        new_state.append(get_units_amount(obs, units.Terran.SensorTower))
        new_state.append(get_units_amount(obs, units.Terran.Bunker))
        new_state.append(get_units_amount(obs, units.Terran.FusionCore))
        new_state.append(get_units_amount(obs, units.Terran.GhostAcademy))
        new_state.append(get_units_amount(obs, units.Terran.Barracks))
        new_state.append(get_units_amount(obs, units.Terran.Factory))
        new_state.append(get_units_amount(obs, units.Terran.Starport))


        #for minimap in obs.feature_minimap:
        #    new_state.extend(minimap.flatten())


        #minimap1 = obs.feature_minimap[0]
        new_state.extend(obs.feature_minimap[2].flatten())      # Feature layer of creep in the minimap (generally will be quite empty, especially on games without zergs hehe)
        new_state.extend(obs.feature_minimap[4].flatten())      # Feature layer of all visible units on the minimap
        final_state = np.array(new_state)
        final_state = np.expand_dims(final_state, axis=0)

        return final_state


    def get_state_dim(self):
        return self._state_size


class Simple64State_1(StateBuilder):
    def build_state(self, obs):
        
        if obs.game_loop[0] == 0:
            command_center = get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

        # Whether or not our supply depot was built
        supply_depot_count = get_units_amount(obs, units.Terran.SupplyDepot)

        # Whether or not our barracks were built
        barracks_count = get_units_amount(obs, units.Terran.Barracks)

        army_count = len(select_army(obs, sc2_env.Race.terran))

        # The supply limit
        supply_limit = obs.player[4]
        # The army supply
        supply_army = obs.player[5]
        # Free supply
        supply_free = get_free_supply(obs)

        # Defining our state, considering our enemies' positions.
        current_state = np.zeros(22)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = supply_army
        current_state[4] = supply_free
        current_state[5] = army_count
        

        # Insteading of making a vector for all coordnates on the map, we'll discretize our enemy space
        # and use a 16x16 grid to store enemy positions by marking a square as 1 if there's any enemy on it.
        hot_squares = np.zeros(16)
        enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        
        for i in range(0, len(enemy_units)):
            y = int(math.ceil((enemy_units[i].x + 1) / 16))
            x = int(math.ceil((enemy_units[i].y + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            # Adds 4 to account for supply_depot_count, barracks_count, supply_limit and army_supply
            current_state[i + 6] = hot_squares[i]
        
        current_state = np.expand_dims(current_state, axis=0)
        return current_state


    def get_state_dim(self):
        return 22