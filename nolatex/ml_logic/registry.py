import os
import time

from colorama import Fore, Style
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save

def save_model(model: keras.Model = None, model_dir: str) -> None:
    """
    Persist trained model locally on the hard drive at f"{model_dir}/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(model_dir, f"{timestamp}.h5")
    save(model_path)

    print(f"✅ Model saved locally at {model_path}")
    return None


def load_model(model_path: str) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    Return None (but do not Raise) if no model is found

    """
    print(Fore.BLUE + f"\nLoading latest model from {model_path}..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    # local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    # local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not model_path:
        return None

    # most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    # print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    # TODO Double-check correctload function
    latest_model = load_model(model_path)

    print("✅ Model loaded from local disk")

    return latest_model
