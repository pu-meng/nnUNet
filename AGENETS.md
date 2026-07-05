# Project instructions for Codex

You are working on PuMengYu's nnUNet / medical image segmentation project.

## Language
- Prefer Chinese explanations.
- Keep answers concise and structured.

## Environment
- The user is using a Windows local computer with VS Code Remote - SSH to connect to a remote Linux server.
- Treat all visible project paths such as `/home/PuMengYu/nnUNet` and `/home/PuMengYu/nnUNet_workspace` as paths on the remote Linux server, not Windows local paths.
- Run shell commands for this project in the remote Linux environment, using Linux paths and Linux command syntax.
- Do not suggest Windows CMD/PowerShell commands unless the user explicitly asks for local Windows-side operations.
- If discussing files opened in VS Code, remember that VS Code is running locally on Windows but editing files through the remote SSH workspace.

## Safety
- Do not run long training commands unless explicitly asked.
- Do not delete, move, or overwrite datasets, checkpoints, result folders, or logs.
- Do not use rm, mv, chmod, chown, git reset, git clean, or pip/conda install unless explicitly asked.
- Do not access network unless explicitly asked.

## Read-only commands
For analysis, freely use safe read-only commands:
- ls
- find
- rg
- grep
- sed -n
- cat
- head
- tail
- pwd
- tree
- git status
- git diff

## Editing rules
- Before editing files, first explain the modification plan.
- Make minimal changes.
- After editing, summarize changed files and key diff.
- When creating or generating any file, always provide the complete absolute Linux path to the final file in the final answer.
- Prefer creating backup files only when modifying important scripts.

## Project focus
- Focus on Dataset003_Liver.
- Focus on MedNeXt, MedNeXt_SizeOV4, MLA/MoE, DeepPlainResGN, DeepDWIBResGN.
- Focus on internal test, external validation, report generation, and experiment summary.
- Key research goal: improve tumor Dice and build a publishable experiment story.


# Response style

When answering, always use clear sections:

## Final Answer
Give the direct conclusion first.

## Files Checked
List files read or edited.

## Evidence
List key metrics, paths, or command outputs.

## Next Steps
Give concise suggested next actions.

Do not mix tool logs with final conclusions.
Do not include raw exploration logs unless they are necessary.
