"""

Particle Swarm Optimization
==================================

(Word-level Textual Adversarial Attacking as Combinatorial Optimization)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import ParticleSwarmOptimization
from textattack.transformations import WordSwapHowNet
from textattack.transformations import WordSwapWordNet

from .attack_recipe import AttackRecipe


class PSOZang2020(AttackRecipe):
    """Zang, Y., Yang, C., Qi, F., Liu, Z., Zhang, M., Liu, Q., & Sun, M.
    (2019).

    Word-level Textual Adversarial Attacking as Combinatorial Optimization.

    https://www.aclweb.org/anthology/2020.acl-main.540.pdf

    Methodology description quoted from the paper:

    "We propose a novel word substitution-based textual attack model, which reforms
    both the aforementioned two steps. In the first step, we adopt a sememe-based word
    substitution strategy, which can generate more candidate adversarial examples with
    better semantic preservation. In the second step, we utilize particle swarm optimization
    (Eberhart and Kennedy, 1995) as the adversarial example searching algorithm."

    And "Following the settings in Alzantot et al. (2018), we set the max iteration time G to 20."
    """

    @staticmethod
    def build(model_wrapper, product=False):
        #
        # Swap words with their synonyms extracted based on the HowNet.
        #
        transformation = WordSwapHowNet()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        #
        # Use untargeted classification for demo, can be switched to targeted one
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Perform word substitution with a Particle Swarm Optimization (PSO) algorithm.
        #
        search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20, product=product)

        return Attack(goal_function, constraints, transformation, search_method)

# same with original recipe.
class PSOZang2020Premise(AttackRecipe):
    @staticmethod
    def build(model_wrapper, product=False):
        transformation = WordSwapHowNet()
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20, product=product)

        return Attack(goal_function, constraints, transformation, search_method)

class PSOZang2020Hypothesis(AttackRecipe):
    @staticmethod
    def build(model_wrapper, product=False):
        transformation = WordSwapHowNet()
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20, product=product)

        return Attack(goal_function, constraints, transformation, search_method)

class PSOZang2020Wordnet(AttackRecipe):
    @staticmethod
    def build(model_wrapper, product=False):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        goal_function = UntargetedClassification(model_wrapper)
        search_method = ParticleSwarmOptimization(pop_size=2, max_iters=2, product=product)
        return Attack(goal_function, constraints, transformation, search_method)