import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyDangerDetector:

    pedestrian_tl = ['pedestrian_green', 'pedestrian_off', 'pedestrian_red']
    crosswalks = ['crosswalk', 'dashed_crosswalk']

    def __init__(self):
        # Antecedent/Consequent objects.
        growth = ctrl.Antecedent(np.arange(0, 11, 1), 'growth')
        position = ctrl.Antecedent(np.arange(0, 11, 1), 'position')
        danger = ctrl.Consequent(np.arange(0, 11, 1), 'danger')

        # Membership functions.
        growth.automf(names=['fast_negative', 'negative', 'neutral', 'positive', 'fast_positive'])
        position['left'] = fuzz.gaussmf(position.universe, 0, 1)
        position['center'] = fuzz.gaussmf(position.universe, 5, 2.5)
        position['right'] = fuzz.gaussmf(position.universe, 10, 1)
        danger.automf(names=['very_low', 'low', 'medium', 'high', 'very_high'])

        # Crosswalk rules
        rule1 = ctrl.Rule(growth['fast_positive'], danger['very_high'])
        rule2 = ctrl.Rule(growth['positive'], danger['high'])
        rule3 = ctrl.Rule(growth['neutral'], danger['low'])
        rule4 = ctrl.Rule(growth['negative'], danger['very_low'])
        rule5 = ctrl.Rule(growth['fast_negative'], danger['very_low'])
        rule6 = ctrl.Rule(position['center'], danger['very_high'])
        rule7 = ctrl.Rule(position['left'], danger['medium'])
        rule8 = ctrl.Rule(position['right'], danger['medium'])

        # Create the control system.
        crosswalk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
        self.crosswalk_simulation = ctrl.ControlSystemSimulation(crosswalk_ctrl)

        # Pedestrian Traffic light rules.
        rule1 = ctrl.Rule(growth['fast_positive'], danger['very_high'])
        rule2 = ctrl.Rule(growth['positive'], danger['high'])
        rule3 = ctrl.Rule(growth['neutral'], danger['low'])
        rule4 = ctrl.Rule(growth['negative'], danger['very_low'])
        rule5 = ctrl.Rule(growth['fast_negative'], danger['very_low'])

        # Create the control system.
        pedestrian_traffic_light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
        self.pedestrian_traffic_light_simulation = ctrl.ControlSystemSimulation(pedestrian_traffic_light_ctrl)

    def get_danger(self, danger_type, growth, position):
        if danger_type in self.pedestrian_tl:
            danger_simulation = self.pedestrian_traffic_light_simulation
        else:
            danger_simulation = self.crosswalk_simulation
        # Inputs.
        danger_simulation.input['growth'] = growth
        danger_simulation.input['position'] = position

        # Compute the output.
        danger_simulation.compute()

        return danger_simulation.output['danger']
