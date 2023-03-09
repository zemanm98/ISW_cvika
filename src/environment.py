import os
from collections import deque
from itertools import chain
from time import perf_counter

import numpy as np
from scipy.stats import truncnorm

from src import graphics as svg
import networkx as nx

from src.glicko import Glicko2
from src.settings import *


class Environment:
    """
    Represents a general environment.
    Very important methods: `act` & `set_to_initial_state`.

    Attributes:

    - _name :class:`str` --> Environment name (for log identification purposes)
    - _state_observable :class:`numpy.ndarray` --> Current observable state of the environment. Has shape equal to `self.observation_shape()`.
    - _state_internal :class:`numpy.ndarray` --> Current internal state of the environment. Has shape equal to `self.internal_shape()`.
    - _state_counter :class:`int` --> The current timestep (amount of `act` calls since calling `set_to_initial_state`).
    - _reward :class:`float` --> Current reward awarded for previous `act` call.
    - _valid_actions :class:`set[int]` --> A set of valid actions corresponding to the current state.
    - _is_terminal :class:`bool` --> A flag indicating whether the current state is terminal or not (True = terminal).
    - __saved_state :class:`tuple` --> A saved tuple of variables fully representing this environment's saved state (see `self.save()`).
    - __cumulative_reward :class:`float` --> Cumulative reward achieved in the current episode (as of the current timestep).
    """

    def __init__(self, name="unk"):
        self._name = name
        self._state_observable = None
        self._state_internal = None
        self._state_counter: int = 0
        self._reward: float = None
        self._valid_actions: set = None
        self._is_terminal: bool = None
        self.__saved_state = None
        self.__cumulative_reward = 0

    def act(self, a) -> tuple[np.ndarray, float, set[int], bool]:
        """
        Perform a single step in the environment given a supplied action.
        :param a: action to take (should be valid, not always checked for performance reasons)
        :return:
        1. New observable state
        2. Reward for taking the action `a` in the previous state and resulting in the new state <-- R(x, a, x')
        3. Set of valid actions in the new state
        4. Whether the new state is terminal or not (True = terminal)
        """
        # logger.debug("taking action %d" % a)
        self._update(a)
        self._state_counter += 1
        self.__cumulative_reward += self._reward
        return np.array(self._state_observable), self._reward, set(self._valid_actions), self._is_terminal

    def set_to_initial_state(self) -> tuple[np.ndarray, float, set[int], bool]:
        """
        Restarts the environment to the initial state.
        :return:
        1. Initial observable state
        2. Initial reward (for our purposes always zero, but can be non-zero [bias, normalization])
        3. Set of valid actions in the initial state
        4. Whether the initial state is terminal or not (True = terminal) (for our purposes always False)
        """
        # logger.debug("resetting")
        self._restart()
        self._state_counter = 0
        self.__cumulative_reward = 0
        return self._state_observable, self._reward, self._valid_actions, self._is_terminal

    def save(self):
        """
        Saves the environment's state into a single "memory cell" (that is, you cannot -- and should not need -- to save
        more than one state at one time) for future restoration. Every call to `save` overwrites the previously saved
        state.

        Useful for implementing AlphaZero and other tree-traversal-based agents.
        """
        self.__saved_state = [np.copy(self._state_observable), np.copy(self._state_internal), self._state_counter,
                              self._reward, set(self._valid_actions), self._is_terminal, self.__cumulative_reward]
        # logger.debug("saved state as %s" % str(self.__saved_state))

    def restore(self):
        """
        Restores the state saved by calling `self.save()`.

        Useful for implementing AlphaZero and other tree-traversal-based agents.
        """
        # logger.debug("restoring state")
        so, si, sc, r, va, t, cr = self.__saved_state
        self._state_observable = np.copy(so)
        self._state_internal = np.copy(si)
        self._state_counter = sc
        self._reward = r
        self._valid_actions = set(va)
        self._is_terminal = t
        self.__cumulative_reward = cr

    def eval_one_episode(self, agent) -> tuple[int, float]:
        """
        Performs evaluation of the supplied agent on a single episode.
        :param agent: target agent to evaluate
        :return:    1. number of states played
                    2. cumulative reward achieved
        """
        self.set_to_initial_state()
        total_reward = 0
        while not self._is_terminal:
            a = agent.best_action(self._state_observable, self._valid_actions)
            self.act(a)
            # logger.debug(f"{agent._action_distribution} -> {a}")
            total_reward += self._reward

        return self._state_counter, total_reward

    def evaluate(self, agent) -> tuple[bool, float, float]:
        """
        Performs evaluation of the supplied agent on multiple episodes, averages and returns the result.
        :param agent: target agent to evaluate
        :return:    1. whether the agent solved the environment (True if solved and the training can stop)
                    2. average cumulative reward achieved by the agent
                    3. average amount of steps taken (mostly for debug purposes)
        """
        avg_reward = 0.0
        avg_steps = 0.0
        for i in range(1, 11):
            logger.info("evaluating run #%d" % i)
            t, r = self.eval_one_episode(agent)
            avg_reward += r
            avg_steps += t

        avg_reward /= 10
        avg_steps /= 10

        if avg_reward < self.target_total_reward:
            logger.info(f"agent failed to solve env, got reward {avg_reward}, "
                        f"expected at least {self.target_total_reward}")
            return False, avg_reward, avg_steps
        else:
            logger.info(f"agent solved env, got reward {avg_reward}")
            return True, avg_reward, avg_steps

    def render_episode(self, agent) -> tuple[int, float]:
        """
        Renders a single episode of this environment performed by a supplied agent into a SVG file (output.svg).
        :param agent: target agent to play the episode
        :return:    1. episode length
                    2. cumulative reward achieved
        """
        logger.info(f"Rendering episode of {self} played by {agent} to {os.getcwd()}")
        self.set_to_initial_state()
        frames = []

        canvas = self.render_current_state()
        total_reward = 0
        counter = 0

        while not self._is_terminal:
            canvas.translate(ACTIONS_X1, ACTIONS_Y1)
            a = agent.render_best_action(self._state_observable, self._valid_actions, canvas)
            self.act(a)
            counter += 1
            total_reward += self._reward
            frames.append(canvas)
            canvas = self.render_current_state()

        canvas.translate(ACTIONS_X1, ACTIONS_Y1)
        self.render_terminal(canvas)
        frames.append(canvas)

        animator = svg.SVGAnimator(WIDTH, HEIGHT, RENDER_PRECISION)
        animator.build_full(frames, "output.svg", self.animator_render_fps, delay=2.0)

        logger.info("Done.")

        return counter, total_reward

    def render_current_state(self) -> svg.Frame:
        """
        Creates a canvas, renders the current state onto it, and returns it.
        :return: canvas with the current state rendered on it
        """
        canvas = self.get_default_canvas()
        canvas.translate(ENV_X1, ENV_Y1)
        self.render_state(canvas)
        canvas.clear_transform()
        canvas.translate(STATUS_X1, STATUS_Y1)
        self.render_status(canvas)
        canvas.clear_transform()
        if self._is_terminal:
            canvas.translate(ACTIONS_X1, ACTIONS_Y1)
            self.render_terminal(canvas)
        return canvas

    def __str__(self):
        return self._name

    def render_terminal(self, canvas: svg.Frame):
        """
        Renders the "state is terminal" message.
        :param canvas: rendering canvas
        """
        canvas.stroke_weight(1)
        canvas.fill("black")
        canvas.text("Terminal state, r = %.2f" % self.__cumulative_reward, ACTIONS_WIDTH * 0.5, ACTIONS_HEIGHT * 0.5,
                    20)

    @staticmethod
    def get_default_canvas() -> svg.Frame:
        """
        Builds a rendering canvas.
        :return: a new canvas for rendering
        """
        canvas = svg.Frame()

        canvas.stroke_weight(BORDER)
        canvas.stroke("black")
        canvas.polyline(((0, ENVSPACE),
                         (WIDTH, ENVSPACE),
                         (WIDTH, 0),
                         (0, 0),
                         (0, HEIGHT),
                         (WIDTH, HEIGHT),
                         (WIDTH, ENVSPACE)))
        canvas.stroke_weight(STATUS_BORDER)
        canvas.stroke_dasharray("6 4")
        canvas.rectangle(STATUS_X1, STATUS_Y1, STATUS_WIDTH, STATUS_HEIGHT)
        canvas.stroke_dasharray(None)

        return canvas

    @property
    def internal_shape(self) -> tuple[int, ...]:
        """
        Shape of the internal state (numpy.ndarray).
        :return: tuple of dimension sizes
        """
        raise NotImplementedError("Must be implemented.")

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """
        Shape of the observable state (numpy.ndarray).
        :return: tuple of dimension sizes
        """
        raise NotImplementedError("Must be implemented.")

    @property
    def num_actions(self) -> int:
        """
        Number of possible (integer) actions an agent can take in the environment.
        Values equal to or larger than this number (or lower than zero) are invalid.
        :return: N if all actions that can be taken are {0, 1, ..., N-1}
        """
        raise NotImplementedError("Must be implemented.")

    @property
    def target_total_reward(self) -> float:
        """
        The reward or rating which, when surpassed by an agent consistently (average of ten tries), that agent is
        considered to have solved the environment.
        :return: float value
        """
        raise NotImplementedError("Must be implemented.")

    @property
    def animator_render_fps(self) -> float:
        """
        Target FPS for when an episode in the environment is rendered using self.render_episode(agent).
        :return: integer value
        """
        raise NotImplementedError("Must be implemented.")

    @property
    def bindings(self) -> tuple[tuple[str, int], ...]:
        """
        A tuple of tuples (char, int) representing which keyboard keys correspond to which actions in the Simulator.
        :return: bindings, tuple of tuples (char, int)
        """
        raise NotImplementedError("Must be implemented.")

    @property
    def action_help_text(self) -> list[str]:
        """
        A list of action descriptions, such as ["go left", "go up", "go down", etc.]
        :return: list of action descriptions
        """
        raise NotImplementedError("Must be implemented.")

    def _restart(self):
        """
        Performs the restart procedure in the environment. Sets the current state to the initial state.
        Must set state, reward, va, term.
        """
        raise NotImplementedError("Must be implemented.")

    def _update(self, a):
        """
        Performs the update procedure in the environment (a single step). Sets the current state to the next state.
        Must set state, reward, va, term.
        """
        raise NotImplementedError("Must be implemented.")

    def render_state(self, canvas):
        """
        Defines how the environment's state should be rendered.
        :param canvas: target canvas to render to
        """
        raise NotImplementedError("Must be implemented.")

    def render_status(self, canvas):
        """
        Defines how the environment's status window (timestep, current reward, state description, etc.)
        should be rendered.
        :param canvas: target canvas to render to
        """
        raise NotImplementedError("Must be implemented.")


class RealTimeEnvironment:
    """
    A special environment which humans would consider real-time, as opposed to turn-based (which is the default).
    Technically, this means that the human playing is not always prompted for an action, but instead if they fail to
    choose an action in a specific time (delta t), a default action is chosen for them.
    """

    @property
    def idle_action(self) -> int:
        """
        Default action to take when the human player does not specify any action in time.
        :return: integer value
        """
        raise NotImplementedError("Must be implemented.")


class RatedAgentWrapper:
    """
    A wrapper class for frozen agents evaluated during AdversarialEnvironment training. Keeps track of agent
    performances and ratings.
    """
    BASE_DIFF = 100
    INIT_RATING = 100
    INIT_RATING_SD = 10
    MIN_RATING_SD = 1
    MAX_RATING_CHANGE = 50
    RATINGS_HISTORY_SIZE = 20
    SCORE_HISTORY_SIZE = 5
    GLICKO_OBJ = Glicko2()

    VICTORY = 1
    DRAW = 0
    DEFEAT = -1

    newest_version = -1

    def __init__(self, agent, version):
        self.agent = agent
        self.score_history = deque(maxlen=RatedAgentWrapper.SCORE_HISTORY_SIZE)
        self.rating = RatedAgentWrapper.GLICKO_OBJ.create_rating()
        self.id = version
        self.tournament_log = []

        RatedAgentWrapper.newest_version = version

    def set_rating(self, new_value):
        self.rating = new_value
        # self.rating_history.add(new_value)
        # ratings = self.rating_history.get_ordered()
        # if len(ratings) > 2:
        #     self.rating_sd = np.std(ratings, ddof=len(ratings) - 1)

    def on_tournament_begin(self):
        self.tournament_log.clear()

    def add_result(self, result, rating):
        self.tournament_log.append((result, rating))

    def on_tournament_end(self):
        self.score_history.append(sum(v[0] for v in self.tournament_log))

    def update_rating(self):
        self.set_rating(
            RatedAgentWrapper.GLICKO_OBJ.rate(
                self.rating,
                [(RatedAgentWrapper.GLICKO_OBJ.result_to_score(res), enm_rtg) for res, enm_rtg in self.tournament_log]
            )
        )

    def is_newest(self):
        return self.id == RatedAgentWrapper.newest_version

    def get_score_running_sum(self):
        return sum(self.score_history)

    def __str__(self):
        return f"Version #{self.id} RATING={self.rating} PERF={self.get_score_running_sum()}"

    def __repr__(self):
        return f"Version #{self.id} RATING={self.rating} PERF={self.get_score_running_sum()}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id and self.agent == other.agent


class AdversarialEnvironment(Environment):
    """
    A specific environment which does not have a single agent trying to maximize some reward, but two agents, white
    and black, trying to maximize, resp. minimize the total reward. It is assumed that the reward for any step in an
    AE is zero except for actions which lead to terminal states, in which case the reward is 1 if white won, -1 if
    black won, and 0 if the game resulted in a draw.

    Attributes:

    - __versions_made_for :class:`dict` --> Keeps track of how many versions does every specific trained agent instance have.
    - __groups :class:`dict` --> Keeps track of living versions of each trained agent instance.
    """

    def __init__(self):
        super().__init__()
        self.__versions_made_for = {}
        self.__groups = {}

    def __play_duel(self, white, black):
        """
        Lets two agents perform a single episode in the environment.
        :param white: agent playing first (maximizing reward)
        :param black: agent playing second (minimizing reward)
        :return: match result
        """
        self.set_to_initial_state()
        players = (white, black)

        current_player = 0
        while not self._is_terminal:
            a = players[current_player].agent.best_action(self._state_observable, self._valid_actions)
            self.act(a)
            current_player = 1 - current_player

        return self._reward

    def __play_duel_and_log_results(self, white, black, scores):
        """
        Same as self.__play_duel(...), but it also logs the results to a `scores` dictionary and to agent result logs
        for rating updates.
        :param white: agent playing first (maximizing reward)
        :param black: agent playing second (minimizing reward)
        :param scores: 
        """
        result_1 = self.__play_duel(white, black)
        white.add_result(result_1, black.rating)
        black.add_result(-result_1, white.rating)
        scores[white] += result_1
        scores[black] -= result_1

    def __play_round(self, agents_sorted, scores, not_faced_yet):
        """
        Performs a round in a tournament. Round is a bunch of matches between matched players, where each match
        consists of two duels, so that both agents get to be both white and black.

        :param agents_sorted: matchmaking result, a flattened list of pairings where
         agents_sorted[i] faces agents_sorted[i+1] for i in [0, 2, 4, ...].
         Example: [A1, A2, B1, B2, C1, C2, D1, ...], same letter ~ facing each other in this round
        :param scores: dictionary of scores for tracking results and future rounds matchmaking
        :param not_faced_yet: edges in the (initially full) graph of players which have not faced each other yet. Used
        for matchmaking.
        """

        # if there is an odd number of players
        if len(agents_sorted) % 2 != 0:
            # the last (worst performing) agent is given a free win
            spare = agents_sorted[-1]
            scores[spare] += 1.0
            # and removed from the array temporarily
            agents_sorted = agents_sorted[:-1]

        for a, b in zip(agents_sorted[::2], agents_sorted[1::2]):
            self.__play_duel_and_log_results(a, b, scores)
            self.__play_duel_and_log_results(b, a, scores)
            not_faced_yet[a].remove(b)
            not_faced_yet[b].remove(a)

    def __run_tournament(self, agents_in_tournament, episode):
        """
        Performs a full tournament to assess the quality of trained, frozen agents. Uses swiss matchmaking.
        Limits the total number of players in the group - if the group gets too large, the worst performing agent is
        removed.
        :param agents_in_tournament: frozen agents to play the tournament
        :param episode: episode number for w&b logging
        """
        if len(agents_in_tournament) < 2:
            return

        for agent in agents_in_tournament:
            agent.on_tournament_begin()

        scores = {a: 0.0 for a in agents_in_tournament}
        not_faced_yet = {a: {b for b in agents_in_tournament if b is not a} for a in agents_in_tournament}

        # first round
        agents_sorted = sorted(agents_in_tournament, key=lambda x: x.rating, reverse=True)
        self.__play_round(agents_sorted, scores, not_faced_yet)

        # matchmaking other rounds
        i = 1
        while True:
            if len(agents_sorted) % 2 == 1:
                min_agent_index = min(range(len(agents_sorted)), key=lambda x: scores[agents_sorted[x]])
                temp = agents_sorted[min_agent_index]
                agents_sorted[min_agent_index] = agents_sorted[-1]
                agents_sorted[-1] = temp
                # lowest agent excluded from pairing (auto-win)
                K = len(agents_sorted) - 1
            else:
                K = len(agents_sorted)

            edges = (
                (i, j, np.square(ROUNDS_TOTAL) - np.square(scores[agents_sorted[i]] - scores[agents_sorted[j]]))
                for i in range(K) for j in range(i + 1, K) if agents_sorted[j] in not_faced_yet[agents_sorted[i]]
            )

            g = nx.Graph()
            g.add_weighted_edges_from(edges)

            start = perf_counter()
            pairing_tuples = nx.algorithms.max_weight_matching(g, maxcardinality=True)
            logger.info(f"perfect pairing took {perf_counter() - start} seconds")

            if len(pairing_tuples) < len(agents_in_tournament) // 2:
                logger.info(f"No more pairings exist, finishing tournament early (rounds={i}).")
                break

            new_indices = list(chain(*(t for t in pairing_tuples)))
            agents_sorted = [agents_sorted[i] for i in new_indices]
            logger.info(str([scores[ag] for ag in agents_sorted]))
            self.__play_round(agents_sorted, scores, not_faced_yet)

            i += 1
            if i == ROUNDS_TOTAL:
                break

        for agent in agents_in_tournament:
            agent.on_tournament_end()

        version_weighted_scores = []
        min_score = min(scores.values())
        for k, agent in enumerate(sorted(agents_in_tournament, key=lambda x: x.id)):
            logger.info(f"{agent} -> {scores[agent]}")
            for i in range(int(scores[agent] - min_score)):
                version_weighted_scores.append(k / len(agents_in_tournament))

        living_versions = [x.id for x in agents_in_tournament]

        version_drag = np.sqrt(np.sum(np.square(np.array(living_versions) - RatedAgentWrapper.newest_version)))

    def evaluate(self, agent) -> tuple[bool, float, float, str]:
        """
        Overridden method of agent evaluation for adversarial environments. The evaluation technique used here is an
        extension of the standard comparison with the current-best version, which allows comparison with top-k
        versions for better rating estimates. The results of the tournament (like who is first, second, etc.) don't
        matter, only the results of individual matches (for rating updates).

        In each call, the agent is "frozen" -> a new instance of it is created with exactly the same parameters (policy)
        and assigned an auto-incrementing ID. This is called a version. Then, every version of the agent created up
        until now gets to clash with the other versions in a swiss tournament to assess its playing strength. The best
        version can be used for training the original agent further, while the worst version is removed from the set
        if the set gets too large. Over time, the ratings of all versions stabilize.

        The rating used for our purposes is Glicko2 (read http://www.glicko.net/glicko/glicko2.pdf for all details).
        During matchmaking, we try to find the lowest-cost pairing in a graph of versions with edges between those
        versions who haven't faced each other in this tournament yet (= swiss matchmaking). The cost of a single match
        between versions A and B is (a-b)^2, where a and b are tournament scores of A and B respectively; the cost of
        the full pairing is the sum of costs of all matches. This ensures that similarly performing versions are matched
        together.

        :param agent:   An instance of a TrainableAgent which is currently being trained on this AdversarialEnvironment.
                        This instance is unchanged during this process, a frozen copy of it is created instead.
        :return:        1. whether the best version solved the environment (True means the training can stop)
                        2. newest version rating
                        3. best version rating
                        4. path to the best version (if the training procedure wants to continue updates from the best
                        version instead of the current version)
        """

        versions_path = os.path.join(agent.get_save_dir(), VERSIONS_PATH)

        if agent not in self.__versions_made_for:
            # first evaluate call using this agent instance
            os.makedirs(versions_path, exist_ok=True)
            # first version index is 0
            self.__versions_made_for[agent] = 0
            # empty group of versions is initialized
            self.__groups[agent] = set()

        # get ID for the newest version
        version_ID = self.__versions_made_for[agent]

        save_path = os.path.join(versions_path, f"v_{version_ID}")
        os.mkdir(save_path)
        agent.save_into(save_path)
        self.__versions_made_for[agent] += 1

        # get the set of versions for tournament
        group_of_players: set = self.__groups[agent]

        # freeze the agent
        frozen_agent = agent.load_frozen_from_path(save_path)

        # create the new version and add it to the group
        rated_agent = RatedAgentWrapper(frozen_agent, version_ID)
        group_of_players.add(rated_agent)

        # perform tournament
        self.__run_tournament(group_of_players, agent.episode)

        # update ratings accordingly
        for player in group_of_players:
            player.update_rating()

        best_rated = max(group_of_players, key=lambda x: x.rating)
        worst_rated = min(group_of_players, key=lambda x: x.rating)
        best_perf = max(group_of_players, key=lambda x: x.get_score_running_sum())

        provisional_rating_players = {p for p in group_of_players if p.rating.is_provisional()}
        group_of_players_without_provisionals = group_of_players.difference(provisional_rating_players)

        best_nonprov_rated = None
        if len(group_of_players_without_provisionals) > 0:
            best_nonprov_rated = max(group_of_players_without_provisionals, key=lambda x: x.rating)
            worst_nonprov_rated = min(group_of_players_without_provisionals, key=lambda x: x.rating)

            if len(group_of_players) > MAX_STABLE_VERSIONS_IN_GROUP:
                group_of_players.remove(worst_nonprov_rated)

        self.__groups[agent] = group_of_players.union(provisional_rating_players)

        best = best_rated if best_nonprov_rated is None else best_nonprov_rated

        return best.rating.value() > self.target_total_reward, rated_agent.rating.value(), \
            best.rating.value(), os.path.join(VERSIONS_PATH, f"v_{best.id}")

    def render_status(self, canvas):
        canvas.fill("black")
        canvas.stroke(0.5)
        canvas.text(f"[Playing: {'white' if self.reward_sign(self._state_observable) > 0 else 'black'}]",
                    STATUS_WIDTH * 0.5, STATUS_HEIGHT * 0.3, STATUS_HEIGHT * 0.2)
        canvas.text("t = %03d, r = %d" %
                    (self._state_counter, self._reward),
                    STATUS_WIDTH * 0.5, STATUS_HEIGHT * 0.7, STATUS_HEIGHT * 0.2)

    def reward_sign(self, observation):
        """
        Returns the reward sign given the supplied state (determines player).
        Generally, in state x, agents seek to maximize reward for taking action a times the reward sign.
        :param observation: queried observable state
        :return: 1 if the current player is white, else -1
        """
        raise NotImplementedError("Must be implemented.")

    @property
    def action_help_text(self):
        raise NotImplementedError("Must be implemented.")

    @property
    def internal_shape(self):
        raise NotImplementedError("Must be implemented.")

    @property
    def observation_shape(self):
        raise NotImplementedError("Must be implemented.")

    @property
    def num_actions(self):
        raise NotImplementedError("Must be implemented.")

    @property
    def target_total_reward(self):
        raise NotImplementedError("Must be implemented.")

    @property
    def animator_render_fps(self):
        raise NotImplementedError("Must be implemented.")

    @property
    def bindings(self):
        raise NotImplementedError("Must be implemented.")

    def _restart(self):
        raise NotImplementedError("Must be implemented.")

    def _update(self, a):
        raise NotImplementedError("Must be implemented.")

    def render_state(self, canvas):
        raise NotImplementedError("Must be implemented.")


class SimpleEnvironment(Environment):
    """
    Simple environment for debug purposes. The goal is to hit the 'currently displayed' peg for as many turns in a row
    as possible.
    """

    @property
    def action_help_text(self):
        return "peg 1", "peg 2", "peg 3", "peg 4", "peg 5"

    @property
    def bindings(self):
        return ()

    @property
    def internal_shape(self):
        return 1,

    @property
    def observation_shape(self):
        return 3,

    @property
    def num_actions(self):
        return 5

    @property
    def target_total_reward(self):
        return 20

    @property
    def animator_render_fps(self):
        return 1

    def _restart(self):
        self._state_internal = np.array([0])
        self._state_observable = np.array([0] * 3)
        self._valid_actions = {0, 1, 2, 3, 4}
        self._reward = 0.0
        self._is_terminal = False

    def _update(self, a):
        if a == self._state_observable[0]:
            self._reward = 1.0
        else:
            self._reward = -1.0

        self._state_observable[0] = (a + 1) % 5
        self._state_internal[0] += 1
        self._state_observable[2] = self._state_internal[0]
        if self._state_internal[0] == 20:
            self._is_terminal = True

    def render_state(self, canvas):
        pass

    def render_status(self, canvas):
        pass

    def __init__(self):
        super().__init__()


class RenderTestEnv(Environment):
    """
    Environment for render testing. Useless otherwise.
    """

    @property
    def internal_shape(self):
        return 0,

    @property
    def target_total_reward(self):
        return 0

    @property
    def bindings(self):
        return ()

    @property
    def action_help_text(self):
        return []

    def render_status(self, canvas):
        canvas.stroke("black")
        canvas.fill("black")
        canvas.stroke_weight(1)
        canvas.text("Dummy text :)", STATUS_WIDTH / 2, STATUS_HEIGHT / 2, STATUS_HEIGHT / 4)

    def __init__(self, amount_of_sticks):
        super().__init__()
        self.amount_of_sticks = amount_of_sticks
        self.sticks = None

    @property
    def observation_shape(self):
        return self.amount_of_sticks, 4

    @property
    def num_actions(self):
        return self.amount_of_sticks

    @property
    def animator_render_fps(self):
        return 30

    def _restart(self):
        self.sticks = []
        for i in range(self.amount_of_sticks):
            self.sticks.append(tuple(np.random.randint(100, 300, 4)))

        self._state_observable = np.array(self.sticks)
        self._reward = 0
        self._is_terminal = False
        self._valid_actions = set(range(self.amount_of_sticks))

    def _update(self, a):
        s = self.sticks[a]
        self.sticks[a] = tuple(c + np.random.rand() * 20 - 10 for c in s)

        self._state_observable = np.array(self.sticks)
        self._reward = 0
        self._is_terminal = self._state_counter > 199
        self._valid_actions = set(range(self.amount_of_sticks))

    def render_state(self, canvas):
        canvas.stroke_weight(3)
        canvas.cap("round")
        for s in self.sticks:
            canvas.line(*s)


class FrozenLake(Environment):
    """
    Game of FrozenLake. The goal is to move your character from one corner of a frozen lake to the other. The lake is
    a square grid where each field is either ice or water - ice is slippery and falling into water freezes you to death.
    The grid is generated randomly (bernoulli); for non-default versions, it is not ensured that a path actually exists
    from one corner to the other.
    """
    MAX_STEP = 99
    ICE = 0
    WATER = 1
    END = 2
    START = 3
    PLAYER = 4
    TYPE_AMOUNT = 5

    ACTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))

    @property
    def target_total_reward(self):
        return 1

    @property
    def internal_shape(self):
        # player position - for faster update
        return 2,

    @property
    def observation_shape(self):
        # Each position is one-hot encoded (according to indices called ICE, WATER, etc.)
        # TYPE_AMOUNT exists just to help define the tensor size.
        # For example, if the player is at the start, the corresponding tensor slice (which there are NxN of) would be
        # [1, 0, 0, 1, 1], because there is ice, no water, no end, there is a start, and there is a player.
        # Some random field with water in it (like the bottom left corner in the default variant) would be
        # [0, 1, 0, 0, 0]. You get the idea.

        return self.size, self.size, FrozenLake.TYPE_AMOUNT

    @property
    def num_actions(self):
        return 4

    @property
    def animator_render_fps(self):
        return 5

    @property
    def bindings(self):
        return ('w', 3), ('a', 2), ('s', 1), ('d', 0)

    @property
    def action_help_text(self):
        return 'go right', 'go down', 'go left', 'go up'

    def _restart(self):
        pass

    def __restart_stochastic(self):
        self.__generate_map()

    def __restart_deterministic(self):
        state = np.random.get_state()
        np.random.seed(self.__np_seed)
        self.__generate_map()
        np.random.set_state(state)

    def __generate_map(self):
        start, end = self.start_end_positions[np.random.randint(0, 4)]
        self.__terrain = np.random.choice(np.arange(0, 2), size=(self.size, self.size), p=(self.p_i, 1 - self.p_i))
        self.__terrain[start[0], start[1]] = FrozenLake.ICE
        self.__terrain[end[0], end[1]] = FrozenLake.ICE
        self._state_observable = np.eye(FrozenLake.TYPE_AMOUNT)[self.__terrain]
        self.__terrain[end[0], end[1]] = FrozenLake.END
        self._state_observable[start[0], start[1], FrozenLake.START] = 1
        self._state_observable[start[0], start[1], FrozenLake.PLAYER] = 1
        self._state_observable[end[0], end[1], FrozenLake.END] = 1
        dx = np.sign(end[0] - start[0])
        dy = np.sign(end[1] - start[1])
        s = self.size - 1
        path = np.random.choice([0] * s + [1] * s, s * 2, replace=False)
        current = list(start)
        for i in range(len(path) - 1):
            if path[i] == 0:
                current[0] += dx
            else:
                current[1] += dy

            self._state_observable[current[0], current[1], :] = 0
            self.__make_ice(current)

        self._reward = 0
        self._is_terminal = False
        self._valid_actions = self.valid_actions_dict[start]
        self.__end_pos = end
        self.__start_pos = self._state_internal = start

    def __make_ice(self, current):
        self._state_observable[current[0], current[1], FrozenLake.ICE] = 1
        self.__terrain[current[0], current[1]] = FrozenLake.ICE

    def _update(self, a):
        assert a in self.valid_actions_dict[
            tuple(self._state_internal)], "action %d is not one of the valid actions %s" % \
                                          (a, str(self.valid_actions_dict[self._state_internal]))

        self._state_observable[self._state_internal[0], self._state_internal[1], FrozenLake.PLAYER] = 0
        self.__update_aux(a)
        self._state_observable[self._state_internal[0], self._state_internal[1], FrozenLake.PLAYER] = 1
        self._valid_actions = self.valid_actions_dict[self._state_internal]
        if self._state_counter == FrozenLake.MAX_STEP:
            self._reward = -1
            self._is_terminal = True
        # logger.debug("player_pos:\n %s" % str(self._state_observable[:, :, FrozenLake.PLAYER]))

    def __update_aux(self, a):
        self._state_internal = self._state_internal[0] + FrozenLake.ACTIONS[a][0], \
                               self._state_internal[1] + FrozenLake.ACTIONS[a][1]

        self.__update_aux_array[self.__terrain[self._state_internal]](a)

    def __update_aux_ice(self, a):
        if a in self.valid_actions_dict[self._state_internal] and np.random.uniform(0, 1) < self.p_s:
            self.__update_aux(a)

    def __update_aux_water(self, a):
        self._reward = -1
        self._is_terminal = True

    def __update_aux_end(self, a):
        self._reward = 1
        self._is_terminal = True

    def render_state(self, canvas: svg.Frame):
        ices = np.where(self._state_observable[:, :, FrozenLake.ICE] == 1)
        waters = np.where(self._state_observable[:, :, FrozenLake.WATER] == 1)

        h = ENV_HEIGHT - 20
        w = ENV_WIDTH
        if w > h:
            rect_side_length = h / self.size
            rect_offset = (w - rect_side_length * self.size) / 2, 0
        else:
            rect_side_length = w / self.size
            rect_offset = 0, (h - rect_side_length * self.size) / 2

        canvas.translate(rect_offset[0], rect_offset[1])

        canvas.stroke_weight(4 / self.size)

        canvas.fill("#a5f2f3")
        canvas.stroke("#257ca3")
        self.render_single_type(canvas, rect_side_length, ices)
        canvas.fill("#396d7c")
        canvas.stroke("#396d7c")
        self.render_single_type(canvas, rect_side_length, waters)

        canvas.stroke("red")
        self.render_player(self._state_internal, canvas, rect_side_length, "red")
        canvas.cap("round")
        self.render_flag(self.__start_pos, canvas, rect_side_length, "black", "white")
        self.render_flag(self.__end_pos, canvas, rect_side_length, "black", "green")

        canvas.clear_transform()

    def render_status(self, canvas):
        canvas.stroke("black")
        canvas.fill("black")
        canvas.stroke_weight(0.5)
        canvas.text("t = %03d" % self._state_counter, STATUS_WIDTH / 2, STATUS_HEIGHT / 2, STATUS_HEIGHT / 4)

    @staticmethod
    def render_flag(which, canvas, rect_side_length, stroke, fill):
        canvas.fill(fill)
        canvas.stroke(stroke)
        canvas.polyline(((which[0] * rect_side_length + rect_side_length * 0.3,
                          which[1] * rect_side_length + rect_side_length * 0.8),
                         (which[0] * rect_side_length + rect_side_length * 0.3,
                          which[1] * rect_side_length + rect_side_length * 0.2),
                         (which[0] * rect_side_length + rect_side_length * 0.7,
                          which[1] * rect_side_length + rect_side_length * 0.35),
                         (which[0] * rect_side_length + rect_side_length * 0.3,
                          which[1] * rect_side_length + rect_side_length * 0.5)))

    @staticmethod
    def render_player(which, canvas, rect_side_length, fill):
        canvas.fill(fill)
        canvas.rectangle(which[0] * rect_side_length + rect_side_length * 0.4,
                         which[1] * rect_side_length + rect_side_length * 0.4,
                         rect_side_length * 0.2,
                         rect_side_length * 0.2)

    @staticmethod
    def render_single_type(canvas, rect_side_length, tile):
        for i in range(len(tile[0])):
            x = tile[0][i]
            y = tile[1][i]
            canvas.rectangle(x * rect_side_length,
                             y * rect_side_length,
                             rect_side_length,
                             rect_side_length)

    def __init__(self, size=4, ice_probability=0.6, slip_probability=0.3, deterministic=True):
        """
        :param size: size of the lake (N by N square grid, default 4)
        :param ice_probability: probability that a random square is ice (dual to water, default 0.6, resp 0.4 for water)
        :param slip_probability: probability that a slip occurs when entering ice. Slip = you are moved
        one step and your action is repeated. Therefore, slips can chain (with diminishing probability of longer slips).
        :param deterministic: whether the initial state is deterministic or, on restart, the environment should
        generate a new map randomly.
        """
        super().__init__("FL")
        self.size = size
        self.p_i = ice_probability
        self.p_s = slip_probability

        if deterministic:
            # good seed for 4x4
            self.__np_seed = 20
            self._restart = self.__restart_deterministic
        else:
            self._restart = self.__restart_stochastic
        self.start_end_positions = (((0, 0), (self.size - 1, self.size - 1)),
                                    ((self.size - 1, 0), (0, self.size - 1)),
                                    ((0, self.size - 1), (self.size - 1, 0)),
                                    ((self.size - 1, self.size - 1), (0, 0)),)
        self.valid_actions_dict = {}
        for i in range(self.size):
            for j in range(self.size):
                self.valid_actions_dict[(i, j)] = set()
        for i in range(self.size - 1):
            for j in range(self.size):
                self.valid_actions_dict[(i, j)].add(0)
        for i in range(self.size):
            for j in range(self.size - 1):
                self.valid_actions_dict[(i, j)].add(1)
        for i in range(1, self.size):
            for j in range(self.size):
                self.valid_actions_dict[(i, j)].add(2)
        for i in range(self.size):
            for j in range(1, self.size):
                self.valid_actions_dict[(i, j)].add(3)

        self.__end_pos = None
        self.__start_pos = None
        self._state_internal = None
        self.__terrain = None
        self.__update_aux_array = (self.__update_aux_ice, self.__update_aux_water, self.__update_aux_end)


class CartPole(Environment, RealTimeEnvironment):
    """
    Game of CartPole. There are multiple possible goals, but the main goal (and the default goal) is to balance an
    inverted pendulum on a cart by moving the cart on a line. The game is over once the angle between the ground and
    the pendulum becomes small enough, which should be prevented by applying force to the cart in order to keep the
    pendulum steady.

    Settings allow for exploration of different versions of introduced stochasticity as well as different targets/goals.
    """
    TIMESTEP_LIMIT = 10000
    HORIZONTAL_SPAN = 10
    CART_WIDTH = 1 / HORIZONTAL_SPAN * ENV_WIDTH
    CART_HEIGHT = 0.4 * CART_WIDTH
    WHEEL_RADIUS = 0.11 * CART_WIDTH
    GROUND_HEIGHT_RATIO = 0.8
    UNIT_SCALAR = CART_WIDTH
    INITIAL_ANGLE = 0.1
    TERMINAL_ANGLE = np.pi * 0.5

    @property
    def target_total_reward(self):
        return 9990

    @property
    def internal_shape(self):
        return 0,

    @property
    def observation_shape(self):
        # tuple (position, velocity, angle, angular velocity, angular acceleration)
        # This tuple defines the physical state fully, the added acceleration of the cart is
        # determined by current action.
        # In the returned tuple, left & counter-clockwise is negative, right and clockwise is positive.
        return 5,

    @property
    def num_actions(self):
        return 3

    @property
    def animator_render_fps(self):
        return 1 / self.dt

    @property
    def bindings(self):
        return ('a', 0), ('s', 1), ('d', 2)

    @property
    def action_help_text(self):
        return 'accelerate left', 'do nothing', 'accelerate right'

    @property
    def idle_action(self):
        return 1

    def _restart(self):
        pass

    def __restart_deterministic(self):
        # position, velocity, angle, angular velocity, angular acceleration
        self._state_observable = np.zeros(shape=self.observation_shape)

        self._state_observable[2] = CartPole.INITIAL_ANGLE
        self._valid_actions = {0, 1, 2}
        self._reward = 0.0
        self._is_terminal = False

    def __restart_stochastic(self):
        self.__restart_deterministic()
        self._state_observable[0] = np.random.normal(0.0, self.init_noise_sd)

    def _update(self, a):
        pass

    @staticmethod
    def __reward_0(a, x1, x2):
        """
        Target 0:   The goal is to not let the pendulum fall as long as possible.
        :param a: action taken
        :param x1: previous position
        :param x2: new position
        :return: always 1
        """
        return 1

    @staticmethod
    def __reward_1(a, x1, x2):
        """
        Target 1:   The goal is to not let the pendulum fall as long as possible while keeping the cart as close to the
                    starting location as possible.
        :param a: action taken
        :param x1: previous position
        :param x2: new position
        :return: always 1 minus the softplus of the distance from the origin (limit 1 at -inf or inf, limit 0 at origin)
        """
        abs_x = np.abs(x2)
        return 1 - abs_x / (1 + abs_x)

    @staticmethod
    def __reward_2(a, x1, x2):
        """
        Target 2:   The goal is to not let the pendulum fall as long as possible while moving the cart as far as
                    possible to the right.
        :param a: action taken
        :param x1: previous position
        :param x2: new position
        :return: the distance traveled to the right
        """
        return x2 - x1 + 0.2

    @staticmethod
    def __reward_3(a, x1, x2):
        """
        Target 3:   The goal is to not let the pendulum fall as long as possible while keeping the cart as close to the
                    starting location as possible while conserving fuel (by taking action 1 = do nothing).
        :param a: action taken
        :param x1: previous position
        :param x2: new position
        :return: always 1 minus the softplus of the distance from the origin (limit 1 at -inf or inf, limit 0 at origin)
                minus 1 if the action taken is not 1 (do nothing)
        """
        abs_x = np.abs(x2)
        return 1 - abs_x / (1 + abs_x) - (a != 1)

    @staticmethod
    def __reward_4(a, x1, x2):
        """
        Target 4:   The goal is to not let the pendulum fall as long as possible while moving the cart as far as
                    possible to the right while conserving fuel (by taking action 1 = do nothing).
        :param a: action taken
        :param x1: previous position
        :param x2: new position
        :return: the distance traveled to the right minus 1 if the action taken is not 1 (do nothing)
        """
        return (x2 - x1) - (a != 1)

    def __update_common(self, a, dt):
        cos_a = np.cos(self._state_observable[2])
        sin_a = np.sin(self._state_observable[2])

        x1 = self._state_observable[0]
        alpha = self._state_observable[4]
        epsilon = self._state_observable[3]

        result_F = \
            (a - 1) - \
            self.pendulum_mass * self.pendulum_length * alpha * cos_a + \
            self.pendulum_mass * self.pendulum_length * epsilon * epsilon * sin_a

        ax = result_F / self.cart_mass

        self._state_observable[4] = (self.gravity * sin_a - ax * cos_a) / self.pendulum_length
        self._state_observable[3] += self._state_observable[4] * dt
        self._state_observable[2] += self._state_observable[3] * dt
        self._state_observable[1] += ax * dt
        self._state_observable[0] += self._state_observable[1] * dt

        self._reward = self.__reward_fn(a, x1, self._state_observable[0])
        self._is_terminal = np.abs(self._state_observable[2]) > CartPole.TERMINAL_ANGLE or \
                            self._state_counter > CartPole.TIMESTEP_LIMIT

    def __update_deterministic(self, a):
        self.__update_common(a, self.dt)

    def __update_stochastic(self, a):
        self.__update_common(a, self.dt + np.random.exponential(self.expected_dt_noise))

    def render_state(self, canvas):
        canvas.stroke("black")
        canvas.fill("white")
        canvas.stroke_weight(3)
        canvas.line(0, ENV_HEIGHT * CartPole.GROUND_HEIGHT_RATIO, ENV_WIDTH, ENV_HEIGHT * CartPole.GROUND_HEIGHT_RATIO)
        canvas.line(ENV_WIDTH * 0.5,
                    ENV_HEIGHT * CartPole.GROUND_HEIGHT_RATIO,
                    ENV_WIDTH * 0.5,
                    ENV_HEIGHT * (CartPole.GROUND_HEIGHT_RATIO + 0.05))

        cart_center_x = ((self._state_observable[0] + CartPole.HORIZONTAL_SPAN * 0.5) %
                         CartPole.HORIZONTAL_SPAN) * CartPole.UNIT_SCALAR

        canvas.rectangle(cart_center_x - CartPole.CART_WIDTH * 0.5,
                         ENV_HEIGHT * CartPole.GROUND_HEIGHT_RATIO - CartPole.CART_HEIGHT - CartPole.WHEEL_RADIUS * 2,
                         CartPole.CART_WIDTH,
                         CartPole.CART_HEIGHT)

        canvas.ellipse(cart_center_x - CartPole.CART_WIDTH * 0.25,
                       ENV_HEIGHT * CartPole.GROUND_HEIGHT_RATIO - CartPole.WHEEL_RADIUS,
                       CartPole.WHEEL_RADIUS,
                       CartPole.WHEEL_RADIUS)

        canvas.ellipse(cart_center_x + CartPole.CART_WIDTH * 0.25,
                       ENV_HEIGHT * CartPole.GROUND_HEIGHT_RATIO - CartPole.WHEEL_RADIUS,
                       CartPole.WHEEL_RADIUS,
                       CartPole.WHEEL_RADIUS)

        cart_center_y = ENV_HEIGHT * CartPole.GROUND_HEIGHT_RATIO - CartPole.CART_HEIGHT * 0.5 - CartPole.WHEEL_RADIUS * 2

        angle = np.degrees(self._state_observable[2])
        pole_tip_y = cart_center_y - self.pendulum_length * CartPole.UNIT_SCALAR

        canvas.rotate(angle, cart_center_x, cart_center_y)
        canvas.stroke_weight(0.5)
        canvas.line(cart_center_x, cart_center_y,
                    cart_center_x, pole_tip_y)
        canvas.stroke_weight(3)
        canvas.fill("black")
        r = np.sqrt(self.pendulum_mass) / 10
        canvas.ellipse(cart_center_x, pole_tip_y,
                       r * CartPole.UNIT_SCALAR, r * CartPole.UNIT_SCALAR)
        canvas.fill("white")
        canvas.rotate(-angle, cart_center_x, cart_center_y)

    def render_status(self, canvas):
        canvas.fill("black")
        canvas.stroke(0.5)
        canvas.text("[x: %+.4f, vx: %+.4f, \u03b8: %+.4f, \u03b5: %+.4f, \u03b1: %+.4f]," %
                    (self._state_observable[0], self._state_observable[1], self._state_observable[2],
                     self._state_observable[3], self._state_observable[4]),
                    STATUS_WIDTH * 0.5, STATUS_HEIGHT * 0.3, STATUS_HEIGHT * 0.2)
        canvas.text("t = %03d, r = %+.4f" %
                    (self._state_counter, self._reward),
                    STATUS_WIDTH * 0.5, STATUS_HEIGHT * 0.7, STATUS_HEIGHT * 0.2)

    def get_hparams(self):
        return {"init_noise_sd": self.init_noise_sd, "expected_dt_noise": self.expected_dt_noise}

    def __init__(self, pendulum_mass: float = 0.3, cart_mass: float = 0.5, pendulum_length: float = 1.0,
                 gravity: float = 4.0, target: int = 0,
                 dt: float = 0.1, init_noise_sd: float = 0.0, expected_dt_noise: float = 0.0):
        """
        :param pendulum_mass: mass of the pendulum weight (rod is massless) in kilograms (default 1)
        :param cart_mass: mass of the cart (default 0.5)
        :param pendulum_length: length of the pendulum rod in meters (default 1)
        :param gravity: environment gravity in meters over second squared (default 5)
        :param target: environment target (default 0, see `def __reward_N` family for details)
        :param dt: time interval over which the environment updates are integrated, i.e. how much time passes in the
        physical world of CartPole between two calls to `act`
        :param init_noise_sd: the standard deviation of the initial position of the cart as a normally-distributed
        random variable (default 0 = deterministic, initial state is otherwise stochastic for values > 0)
        :param expected_dt_noise: the expected value of the time interval noise DTE as an exponentially-distributed
        random variable: actual time interval = dt + DTE (default 0 = deterministic time interval, next states are
        otherwise stochastic for values > 0)
        """

        super().__init__("CP")
        self.expected_dt_noise = expected_dt_noise
        self.init_noise_sd = init_noise_sd
        self.dt = dt
        self.gravity = gravity
        self.pendulum_length = pendulum_length
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass

        targets = [self.__reward_0, self.__reward_1, self.__reward_2, self.__reward_3, self.__reward_4]
        self.__reward_fn = targets[target]

        if init_noise_sd > 0.0:
            self._restart = self.__restart_stochastic
        else:
            self._restart = self.__restart_deterministic

        if expected_dt_noise > 0.0:
            self._update = self.__update_stochastic
        else:
            self._update = self.__update_deterministic


class MountainCar(Environment, RealTimeEnvironment):
    """
    A game of MountainCar. The goal is to leave a valley using a car by applying acceleration to left and right. The
    friction between the car's wheels and the surface is not great enough to just speed in one direction.
    """
    CAR_WIDTH = 1
    CAR_HEIGHT = 0.4 * CAR_WIDTH
    WHEEL_RADIUS = 0.11 * CAR_WIDTH
    UNIT_SCALAR = CAR_WIDTH
    FLAG_HEIGHT = 1

    @property
    def target_total_reward(self):
        return -150

    @property
    def internal_shape(self):
        return 0,

    @property
    def observation_shape(self):
        # tuple (horizontal position, vertical position, and tangential acceleration caused by gravity)
        return 3,

    @property
    def num_actions(self):
        return 3

    @property
    def animator_render_fps(self):
        return 1 / self.dt

    @property
    def bindings(self):
        return ('a', 0), ('s', 1), ('d', 2)

    @property
    def action_help_text(self):
        return 'accelerate left', 'do nothing', 'accelerate right'

    @property
    def idle_action(self):
        return 1

    def _restart(self):
        pass

    def __restart_deterministic(self):
        self._state_observable = np.zeros(shape=self.observation_shape)

        self._valid_actions = {0, 1, 2}
        self._reward = 0.0
        self._is_terminal = False

    def __restart_stochastic(self):
        self._state_observable = np.zeros(shape=self.observation_shape)

        a, b = -1 / self.init_noise_sd, 1 / self.init_noise_sd
        vt, x = truncnorm.rvs(a, b, size=2)

        self._state_observable[0] = x
        self._state_observable[1] = self.__y(x)
        self._state_observable[2] = vt

        self._valid_actions = {0, 1, 2}
        self._reward = 0.0
        self._is_terminal = False

    def _update(self, a):
        pass

    @staticmethod
    def __reward_0(a):
        """
        Target 0:   The goal is to leave the valley as fast as possible.
        :return: always -1
        """
        return -1

    @staticmethod
    def __reward_1(a):
        """
        Target 1:   The goal is to leave the valley as fast as possible while using as litte fuel as possible.
        :return: always -1 minus 1 if the action taken is not 1 (do nothing)
        """
        return -(a != 1) - 1

    def __update_common(self, a, dt):
        y_prime = self.__y_prime(self._state_observable[0])

        at = (a - 1) * self.car_acceleration - y_prime * self.gravity / np.sqrt(1 + y_prime * y_prime)
        self._state_observable[2] += at * dt
        self._state_observable[0] = self.__x_by_arclength(self._state_observable[0], self._state_observable[2] * dt)
        self._state_observable[1] = self.__y(self._state_observable[0])

        self._is_terminal = abs(self._state_observable[0]) >= abs(self.target_x) or self._state_counter == 199
        self._reward = self.__reward_fn(a)

    def __update_deterministic(self, a):
        self.__update_common(a, self.dt)

    def __update_stochastic(self, a):
        self.__update_common(a, self.dt + np.random.exponential(self.expected_dt_noise))

    def render_state(self, canvas):
        canvas.stroke("black")
        canvas.stroke_weight(0.1)
        canvas.cap("round")

        w_ratio = ENV_WIDTH / self.__world_width
        h_ratio = ENV_HEIGHT / self.__world_height

        if w_ratio > h_ratio:
            canvas.translate((ENV_WIDTH - self.__world_width * h_ratio) * 0.5, 0)
            canvas.scale(h_ratio, -h_ratio)
        else:
            canvas.translate(0, (ENV_HEIGHT - self.__world_height * w_ratio) * 0.5)
            canvas.scale(w_ratio, -w_ratio)

        canvas.translate(self.target_x * 1.1, -self.__world_height)

        points = []
        for x in np.linspace(-self.target_x, self.target_x):
            points.append((x, self.__y(x)))
        canvas.polyline(points)
        angle = np.degrees(np.arctan(self.__y_prime(self._state_observable[0])))
        canvas.rotate(angle, self._state_observable[0], self._state_observable[1])
        canvas.translate(self._state_observable[0], self._state_observable[1])
        canvas.stroke_weight(0.05)
        canvas.fill("white")
        canvas.rectangle(-MountainCar.CAR_WIDTH * 0.5,
                         MountainCar.WHEEL_RADIUS * 2,
                         MountainCar.CAR_WIDTH,
                         MountainCar.CAR_HEIGHT)
        canvas.ellipse(-MountainCar.CAR_WIDTH * 0.25,
                       MountainCar.WHEEL_RADIUS,
                       MountainCar.WHEEL_RADIUS,
                       MountainCar.WHEEL_RADIUS)
        canvas.ellipse(MountainCar.CAR_WIDTH * 0.25,
                       MountainCar.WHEEL_RADIUS,
                       MountainCar.WHEEL_RADIUS,
                       MountainCar.WHEEL_RADIUS)
        canvas.translate(-self._state_observable[0], -self._state_observable[1])
        canvas.rotate(-angle, self._state_observable[0], self._state_observable[1])

        self.__render_flag(canvas, self.target_x)
        self.__render_flag(canvas, -self.target_x)

        canvas.clear_transform()

    def __render_flag(self, canvas, x):
        y = self.__y(x)
        canvas.polyline(((x, y),
                         (x, y + MountainCar.FLAG_HEIGHT),
                         (x + MountainCar.FLAG_HEIGHT * 0.66 * -np.sign(x), y + MountainCar.FLAG_HEIGHT * 0.83),
                         (x, y + MountainCar.FLAG_HEIGHT * 0.5)))

    def render_status(self, canvas):
        canvas.fill("black")
        canvas.stroke(0.5)
        canvas.text("[x: %+.4f, y: %+.4f, vt: %+.4f]," %
                    (self._state_observable[0], self._state_observable[1], self._state_observable[2]),
                    STATUS_WIDTH * 0.5, STATUS_HEIGHT * 0.3, STATUS_HEIGHT * 0.2)
        canvas.text("t = %03d, r = %+.4f" %
                    (self._state_counter, self._reward),
                    STATUS_WIDTH * 0.5, STATUS_HEIGHT * 0.7, STATUS_HEIGHT * 0.2)

    def __y(self, x):
        _x = x / self.b
        return self.a * _x * _x * (1 - _x * _x)

    def __y_prime(self, x):
        b_squared = self.b * self.b
        return 2 * self.a * x * (b_squared - 2 * x * x) / (b_squared * b_squared)

    def __init__(self, slope_height: float = 5.0, slope_length: float = 10.0, car_acceleration: float = 1.0,
                 gravity: float = 5.0, target: int = 0,
                 dt: float = 0.1, init_noise_sd: float = 0.0, expected_dt_noise: float = 0.0):
        """
        The valley is a curve f(x) = A * x^2 - B * x^4.
        :param slope_height: height of the slope in meters (default 5)
        :param slope_length: length of the slope (distance between origin and goal) in meters (default 10)
        :param car_acceleration: acceleration caused by action 0 (accelerate left) or 2 (accelerate right)
        :param gravity: environment gravity in meters over second squared (default 5)
        :param target:
        :param dt:
        :param init_noise_sd:
        :param expected_dt_noise:
        """

        super().__init__("MC")
        self.dt = dt
        self.gravity = gravity
        self.car_acceleration = car_acceleration
        self.a = slope_height * 4
        self.b = slope_length * np.sqrt(2)
        self.init_noise_sd = init_noise_sd
        self.expected_dt_noise = expected_dt_noise
        self.target_x = slope_length

        self.__world_height = slope_height + MountainCar.FLAG_HEIGHT
        self.__world_width = self.target_x * 2.2

        self.__x_by_arclength = None
        self.__compute_arclength()

        targets = [self.__reward_0, self.__reward_1]
        self.__reward_fn = targets[target]

        if init_noise_sd > 0.0:
            self._restart = self.__restart_stochastic
        else:
            self._restart = self.__restart_deterministic

        if expected_dt_noise > 0.0:
            self._update = self.__update_stochastic
        else:
            self._update = self.__update_deterministic

    def __compute_arclength(self):
        dx = 0.01 * self.dt

        true_integral = [0.0]
        for x in np.arange(start=dx, stop=self.target_x + 0.1, step=dx):
            y_prime = self.__y_prime(x)
            true_integral.append(true_integral[-1] + np.sqrt(1 + y_prime * y_prime) * dx)
        true_integral.append(true_integral[-1])

        def get_spline_value(f, _x, max_abs_x):
            _x = np.clip(_x, -max_abs_x, max_abs_x)
            sign = np.sign(_x)
            _x = abs(_x)
            i_true = _x / dx
            i_lo = int(i_true)
            i_hi = i_lo + 1
            return ((i_true - i_lo) * f[i_lo] + (i_hi - i_true) * f[i_hi]) * sign

        true_integral_inverse = [0.0]
        start = 0
        b = 1
        for y in np.arange(start=dx, stop=true_integral[-1], step=dx):
            a = start

            while true_integral[a + b] < y:
                b = min(b * 2, len(true_integral) - a - 2)

            while b - a > 1:
                mid = (a + b) // 2
                if true_integral[start + mid] <= y:
                    a = mid
                else:
                    b = mid

            y1 = true_integral[start + a]
            y2 = true_integral[start + b]

            assert y1 <= y < y2

            x = (a + (y - y1) / (y2 - y1)) * dx

            true_integral_inverse.append(x)
        true_integral_inverse.append(true_integral_inverse[-1])

        def x_by_arclength(x0, darc):
            return get_spline_value(true_integral_inverse,
                                    get_spline_value(true_integral, x0, true_integral_inverse[-1]) + darc,
                                    true_integral[-1])

        self.__x_by_arclength = x_by_arclength


class Connect4(AdversarialEnvironment):
    """
    Game of Connect4. The goal is to be the first player to build a horizontal, vertical or diagonal sequence of four
    stones which are being inserted into 7 columns of height 6. The stones are subject to gravity: they fall down and
    stabilize at the lowest unoccupied spot in the column in which they are being inserted. The game is turn-based:
    every turn, the player whose turn it is chooses a column to insert a stone of his color into. Stones cannot be
    inserted into full columns, and if no more non-full columns exist, the game is a draw.
    """
    WIDTH = 7
    WHITE = 0
    BLACK = 1
    EMPTY = 2

    BALL_CENTER_DISTANCE = 1
    BALL_RADIUS = 0.35

    def __init__(self, height=6):
        """
        For simplicity of binding implementation, width is not a parameter of this environment, however, the height is.
        :param height: height of columns (default 6)
        """
        super().__init__()
        self.height = height
        self.width = Connect4.WIDTH

    def reward_sign(self, observation):
        if observation.shape == self.observation_shape:
            return (observation[0, 0, 3] - 0.5) * 2
        else:
            return (observation[:, 0, 0, 3] - 0.5) * 2

    @property
    def internal_shape(self):
        return self.width,  # heights of columns

    @property
    def observation_shape(self):
        return self.width, self.height, 4

    @property
    def num_actions(self):
        return self.width

    @property
    def target_total_reward(self):
        return 10000

    @property
    def animator_render_fps(self):
        return 5

    @property
    def bindings(self):
        return ('a', 0), ('s', 1), ('d', 2), ('f', 3), ('g', 4), ('h', 5), ('j', 6)

    @property
    def action_help_text(self):
        return 'column 0', 'column 1', 'column 2', 'column 3', 'column 4', 'column 5', 'column 6'

    def _restart(self):
        self._state_observable = np.zeros(self.observation_shape)
        self._state_observable[:, :, Connect4.EMPTY] = 1
        self._state_internal = np.zeros(self.internal_shape, dtype=int)
        self._valid_actions = {0, 1, 2, 3, 4, 5, 6}
        self._reward = 0.0
        self._is_terminal = False

    @staticmethod
    def check_right_bound(v, max_v):
        return v <= max_v

    @staticmethod
    def check_left_bound(v, min_v):
        return v >= min_v

    def check_four_in_direction(self, s, x, y, dx, dy, min_x=0, min_y=0):
        max_x = self.width - 1
        max_y = self.height - 1
        right_size = 0
        left_size = 0

        if dx > 0:
            check_x1 = lambda _x: self.check_right_bound(_x, max_x)
            check_x2 = lambda _x: self.check_left_bound(_x, min_x)
        elif dx < 0:
            check_x1 = lambda _x: self.check_left_bound(_x, min_x)
            check_x2 = lambda _x: self.check_right_bound(_x, max_x)
        else:
            check_x1 = lambda _x: True
            check_x2 = lambda _x: True

        if dy > 0:
            check_y1 = lambda _y: self.check_right_bound(_y, max_y)
            check_y2 = lambda _y: self.check_left_bound(_y, min_y)
        elif dy < 0:
            check_y1 = lambda _y: self.check_left_bound(_y, min_y)
            check_y2 = lambda _y: self.check_right_bound(_y, max_y)
        else:
            check_y1 = lambda _y: True
            check_y2 = lambda _y: True

        for i in range(3):
            new_coords = x + dx * (right_size + 1), y + dy * (right_size + 1)
            if check_x1(new_coords[0]) and check_y1(new_coords[1]) and s[new_coords] == 1:
                right_size += 1
            else:
                break
        for i in range(3 - right_size):
            new_coords = x - dx * (left_size + 1), y - dy * (left_size + 1)
            if check_x2(new_coords[0]) and check_y2(new_coords[1]) and s[new_coords] == 1:
                left_size += 1
            else:
                break

        return right_size + left_size >= 3

    def _update(self, a):
        assert a in self._valid_actions, f"Action {a} is not amongst the valid actions {self._valid_actions}."

        current_player = int(self._state_observable[0, 0, 3])
        self._state_observable[a, self._state_internal[a], current_player] = 1
        self._state_observable[a, self._state_internal[a], Connect4.EMPTY] = 0
        s = self._state_observable[:, :, current_player]
        for dx, dy in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
            if self.check_four_in_direction(s, a, self._state_internal[a], dx, dy):
                self._reward = 1 if self._state_observable[0, 0, 3] == Connect4.WHITE else -1
                self._is_terminal = True

        self._state_internal[a] += 1
        self._state_observable[:, :, 3] = 1 - self._state_observable[0, 0, 3]

        if self._state_internal[a] == self.height:
            self._valid_actions.remove(a)
            if np.sum(self._state_internal) == self.width * self.height:
                self._is_terminal = True

    def render_state(self, canvas):
        w_ratio = ENV_WIDTH / self.width
        h_ratio = ENV_HEIGHT / self.height

        if w_ratio > h_ratio:
            canvas.translate((ENV_WIDTH - self.width * h_ratio) * 0.5, 0)
            canvas.scale(h_ratio, h_ratio)
        else:
            canvas.translate(0, (ENV_HEIGHT - self.height * w_ratio) * 0.5)
            canvas.scale(w_ratio, w_ratio)

        canvas.stroke(None)
        canvas.fill("gray")
        canvas.rectangle(0, 0, 7, 6)

        canvas.stroke("black")
        canvas.stroke_weight(0.1)
        canvas.cap("round")

        for x in range(self.width + 1):
            canvas.line(x, 0, x, self.height)

        for y in range(self.height + 1):
            canvas.line(0, y, self.width, y)

        canvas.stroke(None)
        for x in range(self.width):
            for y in range(self.height):
                if self._state_observable[x, y, Connect4.WHITE] == 1:
                    canvas.fill("white")
                    canvas.ellipse(x + 0.5, self.height - y - 0.5, Connect4.BALL_RADIUS, Connect4.BALL_RADIUS)
                elif self._state_observable[x, y, Connect4.BLACK] == 1:
                    canvas.fill("black")
                    canvas.ellipse(x + 0.5, self.height - y - 0.5, Connect4.BALL_RADIUS, Connect4.BALL_RADIUS)


class TicTacToe(AdversarialEnvironment):
    """
    Game of TicTacToe. The goal is to create a horizontal, vertical or diagonal sequence of three stones on a 3x3 square
    grid. The game is turn-based: every turn, the player whose turn it is chooses an empty square on the 3x3 grid and
    inserts a stone of his own color. If no empty square exists, the game is a draw.
    """
    WHITE = 0
    BLACK = 1
    EMPTY = 2

    BALL_CENTER_DISTANCE = 1
    BALL_RADIUS = 0.35

    def __init__(self):
        """
        This environment has no parameters, because the only solid potential parameters are the grid size and sequence
        size, which must be set correspondingly to the other one (for balance) which is a hard task to automate.
        """
        super().__init__()

    def reward_sign(self, observation):
        if observation.shape == self.observation_shape:
            return (observation[0, 0, 3] - 0.5) * -2
        else:
            return (observation[:, 0, 0, 3] - 0.5) * -2

    @property
    def internal_shape(self):
        return 0,

    @property
    def observation_shape(self):
        return 3, 3, 4

    @property
    def num_actions(self):
        return 9

    @property
    def target_total_reward(self):
        return 10000

    @property
    def animator_render_fps(self):
        return 5

    @property
    def bindings(self):
        return ('b', 0), ('g', 1), ('t', 2), ('n', 3), ('h', 4), ('y', 5), ('m', 6), ('j', 7), ('u', 8)

    @property
    def action_help_text(self):
        return '↙', '←', '↖', '↓', '.', '↑', '↘', '→', '↗'

    def _restart(self):
        self._state_observable = np.zeros(self.observation_shape)
        self._state_observable[:, :, TicTacToe.EMPTY] = 1
        self._valid_actions = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        self._reward = 0.0
        self._is_terminal = False

    @staticmethod
    def check_right_bound(v, max_v):
        return v <= max_v

    @staticmethod
    def check_left_bound(v, min_v):
        return v >= min_v

    def check_three_in_direction(self, s, x, y, dx, dy, min_x=0, min_y=0):
        max_x = 2
        max_y = 2
        right_size = 0
        left_size = 0

        if dx > 0:
            check_x1 = lambda _x: self.check_right_bound(_x, max_x)
            check_x2 = lambda _x: self.check_left_bound(_x, min_x)
        elif dx < 0:
            check_x1 = lambda _x: self.check_left_bound(_x, min_x)
            check_x2 = lambda _x: self.check_right_bound(_x, max_x)
        else:
            check_x1 = lambda _x: True
            check_x2 = lambda _x: True

        if dy > 0:
            check_y1 = lambda _y: self.check_right_bound(_y, max_y)
            check_y2 = lambda _y: self.check_left_bound(_y, min_y)
        elif dy < 0:
            check_y1 = lambda _y: self.check_left_bound(_y, min_y)
            check_y2 = lambda _y: self.check_right_bound(_y, max_y)
        else:
            check_y1 = lambda _y: True
            check_y2 = lambda _y: True

        for i in range(2):
            new_coords = x + dx * (right_size + 1), y + dy * (right_size + 1)
            if check_x1(new_coords[0]) and check_y1(new_coords[1]) and s[new_coords] == 1:
                right_size += 1
            else:
                break
        for i in range(2 - right_size):
            new_coords = x - dx * (left_size + 1), y - dy * (left_size + 1)
            if check_x2(new_coords[0]) and check_y2(new_coords[1]) and s[new_coords] == 1:
                left_size += 1
            else:
                break

        return right_size + left_size >= 2

    def _update(self, a):
        assert a in self._valid_actions, f"Action {a} is not amongst the valid actions {self._valid_actions}."

        current_player = int(self._state_observable[0, 0, 3])
        self._state_observable[a // 3, a % 3, current_player] = 1
        self._state_observable[a // 3, a % 3, Connect4.EMPTY] = 0
        self._valid_actions.remove(a)
        s = self._state_observable[:, :, current_player]
        for dx, dy in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
            if self.check_three_in_direction(s, a // 3, a % 3, dx, dy):
                self._reward = 1 if self._state_observable[0, 0, 3] == TicTacToe.WHITE else -1
                self._is_terminal = True

        self._state_observable[:, :, 3] = 1 - self._state_observable[0, 0, 3]

        if len(self._valid_actions) == 0:
            self._is_terminal = True

    def render_state(self, canvas):
        w_ratio = ENV_WIDTH / 3
        h_ratio = ENV_HEIGHT / 3

        if w_ratio > h_ratio:
            canvas.translate((ENV_WIDTH - 3 * h_ratio) * 0.5, 0)
            canvas.scale(h_ratio, h_ratio)
        else:
            canvas.translate(0, (ENV_HEIGHT - 3 * w_ratio) * 0.5)
            canvas.scale(w_ratio, w_ratio)

        canvas.translate(0, -0.1)

        canvas.stroke(None)
        canvas.fill("gray")
        canvas.rectangle(0, 0, 3, 3)

        canvas.stroke("black")
        canvas.cap("round")
        canvas.stroke_weight(0.1)

        for x in range(4):
            canvas.line(x, 0, x, 3)

        for y in range(4):
            canvas.line(0, y, 3, y)

        canvas.stroke(None)
        for x in range(3):
            for y in range(3):
                if self._state_observable[x, y, Connect4.WHITE] == 1:
                    canvas.fill("white")
                    canvas.ellipse(x + 0.5, 3 - y - 0.5, Connect4.BALL_RADIUS, Connect4.BALL_RADIUS)
                elif self._state_observable[x, y, Connect4.BLACK] == 1:
                    canvas.fill("black")
                    canvas.ellipse(x + 0.5, 3 - y - 0.5, Connect4.BALL_RADIUS, Connect4.BALL_RADIUS)


class LectureExample(Environment):
    """
    Example from the presentation you've seen.
    """
    MAX_STEP = 99

    @property
    def target_total_reward(self):
        return 4.4

    @property
    def internal_shape(self):
        return 0,

    @property
    def observation_shape(self):
        return 1,

    @property
    def num_actions(self):
        return 4

    @property
    def animator_render_fps(self):
        return 5

    @property
    def bindings(self):
        return ('a', 0), ('s', 1), ('d', 2), ('f', 3)

    @property
    def action_help_text(self):
        return 'a1', 'a2', 'a3', 'a4'

    def _restart(self):
        self._state_observable[0] = 1
        self._valid_actions = {a for a in self.transitions[self._state_observable[0]]}
        self._reward = 0
        self._is_terminal = False

    def _update(self, a):
        s = int(self._state_observable[0])

        assert a in self.transitions[s], "action %d is not one of the valid actions %s" % \
                                         (a, str(self.transitions[s]))

        self._state_observable[0] = self.transitions[s][a]
        self._reward = self.rewards[s][a]
        self._valid_actions = {a for a in self.transitions[int(self._state_observable[0])]}

        if len(self._valid_actions) == 0:
            assert self._state_observable[0] == 7
            self._is_terminal = True
        elif self._state_counter == LectureExample.MAX_STEP:
            self._reward = -10
            self._is_terminal = True

    def render_state(self, canvas: svg.Frame):
        canvas.stroke("black")
        points = [
            (40, 100),  # s1
            (120, 180),  # s2
            (160, 85),  # s3
            (90, 271),  # s4
            (200, 255),  # s5
            (240, 168),  # s6
            (280, 63),  # s7
        ]

        canvas.stroke_weight(2)
        canvas.stroke("green")
        canvas.ellipse(points[0][0], points[0][1], 18, 18)
        canvas.stroke("red")
        canvas.ellipse(points[6][0], points[6][1], 18, 18)
        canvas.stroke("black")

        for i, p in enumerate(points[1:6]):
            canvas.ellipse(p[0], p[1], 18, 18)

        canvas.stroke("yellow")
        canvas.fill("yellow")
        canvas.ellipse(points[self._state_observable[0] - 1][0], points[self._state_observable[0] - 1][1], 15, 15)
        canvas.stroke("black")
        canvas.fill("none")

        for i, p in enumerate(points):
            canvas.text(f"s{i + 1}", p[0] - 1, p[1] + 3, 8)

        for p0 in self.transitions:
            for a, p1 in self.transitions[p0].items():
                A = points[p0 - 1]
                B = points[p1 - 1]
                canvas.arrow(A[0], A[1], B[0], B[1], 10, 30, 'blue' if p0 == self._state_observable[0] else 'black',
                             f"{self.action_help_text[a]}({self.bindings[a][0]}) {self.rewards[p0][a]:.1f}")

        # canvas.arrow(40, 100, 280, 63, 10, 30)

    def render_status(self, canvas):
        canvas.stroke("black")
        canvas.fill("black")
        canvas.stroke_weight(0.5)
        canvas.text(f"state = {self._state_observable[0]}", STATUS_WIDTH / 2, STATUS_HEIGHT / 2, STATUS_HEIGHT / 4)

    def __init__(self, size=4, ice_probability=0.6, slip_probability=0.3, deterministic=True):
        super().__init__("LE")
        self.transitions = {
            1: {1: 2, 2: 3},
            2: {0: 3, 2: 4, 3: 5},
            3: {0: 7, 1: 6, 2: 5, 3: 2},
            4: {1: 5, 0: 1},
            5: {2: 6},
            6: {3: 7},
            7: {},
        }
        self.rewards = {
            1: {1: 0.9, 2: -3.0},
            2: {0: -1.6, 2: 0.2, 3: 1.1},
            3: {0: 0.2, 1: 5.2, 2: 1.2, 3: 1.6},
            4: {1: 1.3, 0: -2.0},
            5: {2: 0.0},
            6: {3: 0.0},
            7: {},
        }
        self._state_observable = np.zeros(self.observation_shape, dtype=int)
        self._state_internal = None


class MultiArmedBandit(Environment):
    MAX_STEP = 99

    @property
    def internal_shape(self) -> tuple[int, ...]:
        return self.arm_count, 2

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return 1,

    @property
    def num_actions(self) -> int:
        return self.arm_count

    @property
    def target_total_reward(self) -> float:
        return self.expected_best * 0.99 * self.MAX_STEP

    @property
    def animator_render_fps(self) -> float:
        return 15

    @property
    def bindings(self) -> tuple[tuple[str, int], ...]:
        return tuple((chr(ord('a') + i), i) for i in range(self.arm_count))

    @property
    def action_help_text(self) -> list[str]:
        return [f"press arm {i}" for i in range(self.arm_count)]

    def _restart(self):
        self._valid_actions = {i for i in range(self.arm_count)}
        self._reward = 0
        self._is_terminal = False

    def _update(self, a):
        lo, hi = self._state_internal[a]
        self._reward = np.random.rand() * (hi - lo) + lo
        self._is_terminal = self._state_counter == self.MAX_STEP

    def render_state(self, canvas):
        pass

    def render_status(self, canvas):
        canvas.stroke("black")
        canvas.fill("black")
        canvas.stroke_weight(0.5)
        canvas.text(f"Hit = {self._state_counter}, last reward = {self._reward:.2f}, target = "
                    f"{self.target_total_reward:.0f}",
                    STATUS_WIDTH / 2, STATUS_HEIGHT / 2 + 2, STATUS_HEIGHT / 4)

    def __init__(self, arm_count=10, deterministic_init=False):
        super().__init__()

        if deterministic_init:
            np.random.seed(7)
        else:
            np.random.seed(datetime.datetime.now().microsecond)

        self.arm_count = arm_count
        self._state_observable = np.zeros(self.observation_shape)
        self._state_internal = np.zeros(self.internal_shape)
        for i in range(self.arm_count):
            self._state_internal[i][0] = 10 * np.random.rand()
            self._state_internal[i][1] = 10 * np.random.rand()
        best_idx = np.argmax(np.sum(self._state_internal, axis=1))
        self.expected_best = (self._state_internal[best_idx][0] + self._state_internal[best_idx][1]) * 0.5
