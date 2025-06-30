# from mimic_real.envs.base.clean_env import BaseEnv
# from mimic_real.envs.base.base_env_config import BaseEnvCfg, BaseAgentCfg
# from mimic_real.utils.task_registry import task_registry

# from mimic_real.envs.hi_clean.hi_clean_config import HICleanAgentCfg, HICleanEnvCfg
# task_registry.register("hi_clean", BaseEnv, HICleanEnvCfg(), HICleanAgentCfg())

from mimic_real.envs.mimic.hi_mimic_env import BaseEnv
from mimic_real.utils.task_registry import task_registry
from mimic_real.envs.mimic.hi_mimic_config import HIMimicAgentCfg, HIMimicEnvCfg
task_registry.register("hi_mimic", BaseEnv, HIMimicEnvCfg(), HIMimicAgentCfg())