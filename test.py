from calculate_beta import calculate_beta
from calculate_tau import calculate_tau
from utils.utils import StoreDict, get_model_path
import sys 
from utils import ALGOS, create_test_env, get_saved_hyperparams



seed = 1
algo = 'ppo'

device = 'cuda'



_, model_path, log_path = get_model_path(
    1,
    './trained_models/',
    algo, 
    'RoboschoolHalfCheetah-v1',
    True,
    False,
    False,
)

kwargs = dict(seed=seed)

# Check if we are running python 3.8+
# we need to patch saved model under python 3.6/3.7 to load them
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

custom_objects = {}
if newer_python_version:
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

model = ALGOS[algo].load(model_path, env=None, custom_objects=custom_objects, device=device, **kwargs)



#beta_network = calculate_beta(env = 'RoboschoolHalfCheetah-v1', log='./debug/beta_log/', 
               #file_p='./offline_data/real_world_data.pkl', file_q = './offline_data/sim_world_data.pkl')

calculate_tau(model=model, file_p='./offline_data/real_world_data.pkl', file_q = './offline_data/sim_world_data.pkl', log_path=log_path)

