import os
import openai
from typing import Dict, Any, Optional

class ToTVRGenerator:
    """
    ToT-guided Vulnerability Reasoning (ToT-VR) Process
    Based on the prompt templates defined in the paper.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the ToT-VR generator.
        
        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY environment variable.
            model: The model to use. Defaults to "gpt-4o" (can be set to "gpt-5" when available).
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        
        self.system_prompt = (
            "You are a security analysis assistant specialized in vulnerability reasoning for source code. "
            "Follow the instructions step by step and produce outputs in a professional vulnerability-analysis style. "
            "When multiple candidate reasoning branches are requested, explicitly distinguish them as Branch 1, Branch 2, and Branch 3. "
            "When pruning or verifying branches, base your decision only on the provided code semantics, structural evidence, "
            "and external vulnerability descriptions. Do not generate unnecessary discussion outside the requested format."
        )

    def _call_llm(self, prompt: str) -> str:
        """Helper to call the LLM with the system prompt and a specific user prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def step_1_branch_initialization(self, function_code: str, critical_lines: str) -> str:
        prompt = (
            "Analyze the following code and critical lines. Construct exactly three candidate reasoning branches for the potential vulnerability. "
            "Each branch should represent one possible vulnerability interpretation and include: (1) the possible vulnerability type, and (2) a brief explanation of the trigger point or causal mechanism.\n\n"
            f"Code:\n{function_code}\n\n"
            f"Critical Code Lines:\n{critical_lines}\n\n"
            "Output Format:\n"
            "Branch 1: [possible vulnerability type] -- [brief explanation]\n"
            "Branch 2: [possible vulnerability type] -- [brief explanation]\n"
            "Branch 3: [possible vulnerability type] -- [brief explanation]"
        )
        return self._call_llm(prompt)

    def step_2_branch_reflection_and_pruning(self, candidate_branches: str) -> str:
        prompt = (
            "Review the following candidate reasoning branches. Check whether each branch is internally consistent and aligned with the code semantics. "
            "Revise weak branches if necessary, and prune unsupported branches. Retain only the branches that remain plausible based on the provided code.\n\n"
            f"Candidate Branches:\n{candidate_branches}\n\n"
            "Output Format:\n"
            "Retained Branches: [list of retained branches]\n"
            "Pruned Branches: [list of pruned branches]\n"
            "Reason: [brief explanation of the pruning result]"
        )
        return self._call_llm(prompt)

    def step_3_branch_refinement(self, retained_branches: str, vulnerable_lines: str, line_relations: str) -> str:
        prompt = (
            "Refine the retained branches using the following structural evidence. The vulnerable lines and their relations to the critical code lines "
            "should be used to update the trigger point, dependency path, and causal explanation of each retained branch. "
            "If any retained branch is no longer supported by this structural evidence, prune it.\n\n"
            f"Retained Branches:\n{retained_branches}\n\n"
            f"Vulnerable Lines:\n{vulnerable_lines}\n\n"
            f"Relations to Critical Lines:\n{line_relations}\n\n"
            "Output Format:\n"
            "Refined Retained Branches: [updated branches]\n"
            "Pruned Branches: [if any]\n"
            "Reason: [brief explanation of the refinement result]"
        )
        return self._call_llm(prompt)

    def step_4_branch_verification(self, refined_branches: str, cve_description: str) -> str:
        prompt = (
            "Compare the retained branch with the following official CVE description. Verify whether the retained branch is consistent with the known vulnerability semantics. "
            "If it is not supported, state that the branch should be reconsidered. Otherwise, confirm it as the verified branch.\n\n"
            f"Retained Branch:\n{refined_branches}\n\n"
            f"Official CVE Description:\n{cve_description}\n\n"
            "Output Format:\n"
            "Verified Branch: [branch identifier]\n"
            "Verification Result: [consistent / inconsistent]\n"
            "Reason: [brief explanation]"
        )
        return self._call_llm(prompt)

    def step_5_description_synthesis(self, verified_branch: str) -> str:
        prompt = (
            "Generate a final vulnerability description based only on the verified branch. The description should be a single coherent paragraph "
            "in professional security-analysis style. It must clearly cover the vulnerability type, root cause, trigger condition, and impact. "
            "Do not include branch labels, bullet points, or intermediate reasoning.\n\n"
            f"Verified Branch:\n{verified_branch}\n\n"
            "Output Format: A single paragraph containing the final vulnerability description only."
        )
        return self._call_llm(prompt)

    def generate_vulnerability_description(self, 
                                           function_code: str, 
                                           critical_lines: str, 
                                           vulnerable_lines: str, 
                                           line_relations: str, 
                                           cve_description: str) -> Dict[str, Any]:
        """
        Run the full ToT-VR pipeline and return the intermediate and final results.
        """
        print("Starting ToT-VR Process...")
        
        print("Step 1: Branch Initialization...")
        step1_res = self.step_1_branch_initialization(function_code, critical_lines)
        
        print("Step 2: Branch Reflection and Pruning...")
        step2_res = self.step_2_branch_reflection_and_pruning(step1_res)
        
        print("Step 3: Branch Refinement...")
        step3_res = self.step_3_branch_refinement(step2_res, vulnerable_lines, line_relations)
        
        print("Step 4: Branch Verification...")
        step4_res = self.step_4_branch_verification(step3_res, cve_description)
        
        print("Step 5: Description Synthesis...")
        final_description = self.step_5_description_synthesis(step4_res)
        
        print("ToT-VR Process Completed.")
        
        return {
            "step_1_initialization": step1_res,
            "step_2_reflection_and_pruning": step2_res,
            "step_3_refinement": step3_res,
            "step_4_verification": step4_res,
            "final_description": final_description
        }

# Example usage
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment before running
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    try:
        # Assuming gpt-5 is available, otherwise defaults to gpt-4o
        generator = ToTVRGenerator(model="gpt-5")
        
        sample_code = '''
char tmp[64];
strcpy(tmp, buf);
printf("%s", tmp);
'''
        sample_critical_lines = "char tmp[64];\nstrcpy(tmp, buf);"
        sample_vulnerable_lines = "strcpy(tmp, buf);"
        sample_line_relations = "Line 2 (strcpy) writes to tmp defined at Line 1 without bounds checking."
        sample_cve_description = "Stack-based buffer overflow caused by unsafe use of strcpy without bounds checking."
        
        results = generator.generate_vulnerability_description(
            function_code=sample_code,
            critical_lines=sample_critical_lines,
            vulnerable_lines=sample_vulnerable_lines,
            line_relations=sample_line_relations,
            cve_description=sample_cve_description
        )
        
        print("\n--- Final Generated Description ---")
        print(results["final_description"])
        
    except Exception as e:
        print(f"Error: {e}")
