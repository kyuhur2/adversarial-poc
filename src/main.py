import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Type
from src.lib.base_modifier import BaseModifier
from src.lib.adversarial_perturbation import AdversarialPerturbation
from src.lib.feature_obfuscation import FeatureObfuscation
from src.lib.geometric_distortions import GeometricDistortion

def get_modifier_class(method: str) -> Type[BaseModifier]:
    """Returns the corresponding modifier class based on the method string."""
    modifiers = {
        "perturbation": AdversarialPerturbation,
        "obfuscation": FeatureObfuscation,
        "distortion": GeometricDistortion,
    }
    
    if method in modifiers:
        return modifiers[method]
    else:
        raise ValueError(f"Unknown method: {method}. Available options are: {list(modifiers.keys())}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Apply various image modifications to prevent misuse.")
    
    parser.add_argument("--method", type=str, required=True,
                        choices=["perturbation", "obfuscation", "distortion"],
                        help="Choose the method: perturbation, obfuscation, or distortion.")
    
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input image.")
    
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the modified image.")
    
    args = parser.parse_args()

    # dynamically load the class for the chosen method
    modifier_class: Type[BaseModifier] = get_modifier_class(args.method)
    modifier_instance: BaseModifier = modifier_class()

    # apply the chosen modification method
    modifier_instance.apply(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
