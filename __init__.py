import os
import json
import random
import re

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
POSITIVE_SAVE_FILE = os.path.join(CURRENT_DIR, "saved_prompts_positive.json")
NEGATIVE_SAVE_FILE = os.path.join(CURRENT_DIR, "saved_prompts_negative.json")

def load_prompts(filepath):
    """Load prompts from JSON file"""
    prompts = ["None"]
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prompts = ["None"] + list(reversed(data))
        except Exception as e:
            print(f"Error loading prompts from {filepath}: {e}")
    return prompts

def save_prompt(filepath, text):
    """Save a new prompt to JSON file"""
    current_data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
        except:
            current_data = []
    
    if text not in current_data:
        current_data.append(text)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=4)
        print(f"✓ Saved: {text[:50]}...")
        return True
    return False

def delete_prompt(filepath, text):
    """Delete a prompt from JSON file"""
    if not os.path.exists(filepath) or text == "None":
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        
        if text in current_data:
            current_data.remove(text)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, indent=4)
            print(f"✗ Deleted: {text[:50]}...")
            return True
    except Exception as e:
        print(f"Error deleting prompt: {e}")
    return False

class UltimatePromptManager:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        positive_prompts = load_prompts(POSITIVE_SAVE_FILE)
        negative_prompts = load_prompts(NEGATIVE_SAVE_FILE)

        return {
            "required": {
                "clip": ("CLIP",),
                
                # ===== POSITIVE PROMPT SECTION =====
                "positive_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Main Positive Prompt..."}),
                "positive_library": (positive_prompts, ),
                "positive_mode": (["Combine: Main + Saved", "Combine: Saved + Main", "Use Main Only", "Use Saved Only", "RANDOM Saved"],),
                "positive_prefix": ("STRING", {"multiline": False, "default": "", "placeholder": "Prefix (e.g. masterpiece, best quality)"}),
                "positive_suffix": ("STRING", {"multiline": False, "default": "", "placeholder": "Suffix (e.g. 8k, detailed)"}),
                
                # ===== NEGATIVE PROMPT SECTION =====
                "negative_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Main Negative Prompt..."}),
                "negative_library": (negative_prompts, ),
                "negative_mode": (["Combine: Main + Saved", "Combine: Saved + Main", "Use Main Only", "Use Saved Only", "RANDOM Saved"],),
                "negative_prefix": ("STRING", {"multiline": False, "default": "", "placeholder": "Negative Prefix"}),
                "negative_suffix": ("STRING", {"multiline": False, "default": "", "placeholder": "Negative Suffix"}),
                
                # ===== MANAGEMENT =====
                "save_positive": (["No", "Yes - Save Positive"],),
                "save_negative": (["No", "Yes - Save Negative"],),
                "delete_mode": (["OFF", "Delete Selected Positive", "Delete Selected Negative"],),
            },
            "optional": {
                "find_text": ("STRING", {"multiline": False, "default": "", "placeholder": "Find word..."}),
                "replace_with": ("STRING", {"multiline": False, "default": "", "placeholder": "Replace with..."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING",)
    RETURN_NAMES = ("POSITIVE", "NEGATIVE", "POSITIVE_TEXT", "NEGATIVE_TEXT",)
    FUNCTION = "process_prompts"
    CATEGORY = "My Custom Nodes"

    def process_prompts(self, clip, positive_text, positive_library, positive_mode, positive_prefix, positive_suffix,
                       negative_text, negative_library, negative_mode, negative_prefix, negative_suffix,
                       save_positive, save_negative, delete_mode, find_text="", replace_with=""):
        
        # ===== DELETE MODE LOGIC =====
        if delete_mode == "Delete Selected Positive":
            if delete_prompt(POSITIVE_SAVE_FILE, positive_library):
                print("⚠ REFRESH YOUR BROWSER (F5) to update the dropdown!")
        
        if delete_mode == "Delete Selected Negative":
            if delete_prompt(NEGATIVE_SAVE_FILE, negative_library):
                print("⚠ REFRESH YOUR BROWSER (F5) to update the dropdown!")

        # ===== SAVE LOGIC =====
        if save_positive == "Yes - Save Positive" and positive_text.strip():
            if save_prompt(POSITIVE_SAVE_FILE, positive_text):
                print("⚠ REFRESH YOUR BROWSER (F5) to see new positive prompt!")

        if save_negative == "Yes - Save Negative" and negative_text.strip():
            if save_prompt(NEGATIVE_SAVE_FILE, negative_text):
                print("⚠ REFRESH YOUR BROWSER (F5) to see new negative prompt!")

        # ===== PROCESS POSITIVE =====
        final_positive = self._build_prompt(
            positive_text, positive_library, positive_mode, 
            positive_prefix, positive_suffix, find_text, replace_with, POSITIVE_SAVE_FILE
        )

        # ===== PROCESS NEGATIVE =====
        final_negative = self._build_prompt(
            negative_text, negative_library, negative_mode,
            negative_prefix, negative_suffix, find_text, replace_with, NEGATIVE_SAVE_FILE
        )

        print(f"🟢 POSITIVE: {final_positive}")
        print(f"🔴 NEGATIVE: {final_negative}")

        # ===== ENCODE =====
        tokens_pos = clip.tokenize(final_positive)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)

        tokens_neg = clip.tokenize(final_negative)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)

        return (
            [[cond_pos, {"pooled_output": pooled_pos}]], 
            [[cond_neg, {"pooled_output": pooled_neg}]],
            final_positive,
            final_negative
        )

    def _build_prompt(self, text_main, saved_library, mode, prefix, suffix, find_text, replace_with, save_file):
        """Helper function to build a prompt with all modifiers"""
        prompt_content = ""
        saved_text = "" if saved_library == "None" else saved_library

        # Handle Random Mode
        if mode == "RANDOM Saved":
            if os.path.exists(save_file):
                try:
                    with open(save_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data:
                            prompt_content = random.choice(data)
                        else:
                            prompt_content = text_main
                except:
                    prompt_content = text_main
            else:
                prompt_content = text_main
        elif mode == "Use Main Only":
            prompt_content = text_main
        elif mode == "Use Saved Only":
            prompt_content = saved_text
        elif mode == "Combine: Main + Saved":
            sep = ", " if text_main and saved_text else ""
            prompt_content = f"{text_main}{sep}{saved_text}"
        elif mode == "Combine: Saved + Main":
            sep = ", " if text_main and saved_text else ""
            prompt_content = f"{saved_text}{sep}{text_main}"

        # Apply Find & Replace
        if find_text != "":
            prompt_content = prompt_content.replace(find_text, replace_with)

        # Assemble with Prefix and Suffix
        parts = [prefix, prompt_content, suffix]
        parts = [p for p in parts if p.strip()]
        final_string = ", ".join(parts)

        # Cleanup
        final_string = re.sub(r'\s+', ' ', final_string)
        final_string = re.sub(r',\s*,', ',', final_string)
        final_string = final_string.strip(', ')

        return final_string

NODE_CLASS_MAPPINGS = {
    "UltimatePromptManager": UltimatePromptManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimatePromptManager": "🎨 Ultimate Prompt Manager"
}