from __future__ import annotations

import uuid

from src import graphics as svg
from src.environment import Environment
from src.settings import *


class Agent:
    """
    Represents a general agent. The important method is `best_action(...)`, which determines the agent's policy.
    """

    def __init__(self, env: Environment):
        """
        Gives an environment to the agent for setup purposes. Agent instances, once created, cannot be generally
        recycled for different environments simply because the environment state representation does not match the
        expected size/shape/type, etc., for which the agent was originally prepared.
        :param env: target environment to create a policy for

        Attributes:

        - _env :class:`Environment` --> the environment instance
        - _action_distribution :class:`numpy.ndarray` or :class:`Iterable[int]` --> distribution of action probabilities/values
        - id :class:`uuid.UUID` --> unique ID for this agent

        """
        self._env = env
        self._action_distribution = np.zeros(shape=(env.num_actions,))
        self.id = Agent.make_id()

    def best_action(self, state, valid_actions) -> int:
        """
        The only really important method of an Agent - given a state and set of valid actions, which action is optimal?
        In general environments, this will be the action which maximizes the expected cumulative reward, but in
        adversarial environments, if the state indicates the current player is black, the best action will be that
        which minimizes the expected cumulative reward, that is, the best action is the argmax of the
        action distribution multiplied by the reward sign of the current state (see AdversarialEnvironment).

        This function should always set _action_distribution. It is assumed that in order to compare the quality of
        actions, an agent will always create an "action distribution" and choose an action according to an argmax/min
        of this distribution. If the _action_distribution is not specified, only minor issues should arise, as it is
        only used for logging and visualization, however, its proper specification can speed up your debugging process.

        :param state: environment state
        :param valid_actions: valid actions in this state
        :return: integer specifying the optimal action to take
        """
        raise NotImplementedError("Must be implemented.")

    @classmethod
    def make_id(cls) -> uuid.UUID:
        """
        Creates a unique ID for the agent (logging purposes).
        :return:
        """
        return uuid.uuid4()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def render_best_action(self, state, valid_actions, canvas) -> int:
        """
        Gets the best action from the agent, renders the distribution and returns the action.
        :param state: target state
        :param valid_actions: corresponding valid actions
        :param canvas: canvas to draw the distribution to
        :return: best action
        """
        a = self.best_action(state, valid_actions)
        self.render_action_distribution(canvas, a)
        return a

    def render_action_distribution(self, canvas: svg.Frame, a) -> None:
        """
        Subprocedure of the `render_best_action` function, it renders the action distribution to a specified canvas.
        :param canvas: target canvas to render the distribution to
        :param a: best action according to the distribution
        """
        bin_width = \
            (ACTIONS_WIDTH - 2 * HIST_MARGIN) / (
                    self._env.num_actions + ACTIONS_HISTOGRAM_MARGIN_RATIO * (self._env.num_actions - 1))
        bin_spacing = bin_width * (1 + ACTIONS_HISTOGRAM_MARGIN_RATIO)

        if len(self._action_distribution.shape) == 2:  # batch to single distribution
            distribution = np.array(self._action_distribution[0])
        else:
            distribution = np.array(self._action_distribution)

        # print(self._action_distribution)

        distr_max = np.max(distribution)
        distr_min = np.min(distribution)
        best_action = a

        if distr_max != distr_min:
            distribution /= np.maximum(np.abs(distr_max), np.abs(distr_min))
            distribution = np.clip(distribution, -1.0, 1.0)

        canvas.stroke_weight(bin_width)
        canvas.cap(svg.CAP_BUTT)
        canvas.stroke("#819aff")
        canvas.translate(HIST_MARGIN, 0)
        canvas.scale(1, ACTIONS_HIST_HEIGHT)

        for i in range(self._env.num_actions):
            x = i * bin_spacing + bin_width / 2
            # print(best_action, i)
            if i == best_action:
                canvas.stroke("#ffba81")

            canvas.line(x, 0.5, x, 0.5 - HIST_HEIGHT * distribution[i])

            if i == best_action:
                canvas.stroke("#819aff")

        canvas.stroke("black")
        canvas.stroke_weight(0.015)
        canvas.line(-MARGIN, 0.5, ACTIONS_WIDTH - 2 * HIST_MARGIN + MARGIN, 0.5)
        canvas.stroke_dasharray("10 6")
        canvas.stroke_weight(0.01)
        canvas.line(-MARGIN, 0.5 - HIST_HEIGHT, ACTIONS_WIDTH - 2 * HIST_MARGIN + MARGIN, 0.5 - HIST_HEIGHT)
        canvas.line(-MARGIN, 0.5 + HIST_HEIGHT, ACTIONS_WIDTH - 2 * HIST_MARGIN + MARGIN, 0.5 + HIST_HEIGHT)
        canvas.stroke_dasharray(None)

        canvas.scale(1, 1 / ACTIONS_HIST_HEIGHT)
        canvas.fill("black")
        canvas.stroke_weight(0.2)

        canvas.text("0", -MARGIN - 10, 0.5 * ACTIONS_HIST_HEIGHT, 14)

        for i in range(self._env.num_actions):
            x = i * bin_spacing + bin_width / 2
            canvas.text("%d" % i, x, ACTIONS_HIST_HEIGHT + 20, 12)
            # canvas.text("%.2e" % original_distribution[i], x, ACTIONS_HIST_HEIGHT + 30, 10)

        canvas.clear_transform()


class RandomAgent(Agent):
    """
    An agent choosing random actions as its policy. Good for debugging purposes.
    """

    def best_action(self, state, valid_actions):
        a = np.random.choice(tuple(valid_actions), 1)[0]
        self._action_distribution = np.full(shape=(self._env.num_actions,), fill_value=0.5)
        self._action_distribution[a] = 1

        return a


class BalancingAgent(Agent):
    """
    An agent which seeks to maintain or throw off balance of a certain value in a vector-represented environment.
    Great for solving CartPole and MountainCar.

    BalancingAgent(target_prop=2) solves CartPole() and MountainCar()
    """

    def __init__(self, env, target_prop, span=0.0, lower=0, idle=1, upper=2):
        """
        :param env: some environment with vector represented (rank 1) observable states
        :param target_prop: index of the 'property' to dis/balance, e.g. if [a, b, c] is the state, target_prop = 2
        selects 'c' to be the monitored property
        :param span: size of the "idling" interval, that is, if the property is within [-span, span),
        the idle action is taken
        :param lower: action to take if 'property' is less than '-span'                                 (-inf, -span)
        :param idle: action to take if 'property' is greater or equal to '-span' and less than 'span'   [-span, span)
        :param upper: action to take if 'property' is greater or equal to 'span'                        [span, inf)
        """
        super().__init__(env)
        self.upper = upper
        self.idle = idle
        self.lower = lower
        self.span = span
        self.target_prop = target_prop

    def best_action(self, state, valid_actions):
        if state[self.target_prop] < -self.span:
            a = self.lower
        elif state[self.target_prop] < self.span:
            a = self.idle
        else:
            a = self.upper
        self._action_distribution = np.full(shape=(self._env.num_actions,), fill_value=0.5)
        self._action_distribution[a] = 1

        return a
