@echo off
title PRISM Forensics - One-Click Git Push
cls
echo =====================================================================
echo    PRISM Forensics - One-Click Git Push
echo =====================================================================
echo.
echo Select Member Contribution to Push:
echo   [1] Tanush (Part 1 - Lead Architect, Neural Backbone & Streamlit GUI)
echo   [2] Somil  (Part 2 - Physical, Temporal & Container Vision Agents)
echo   [3] Kulsum (Part 3 - Biometrics, Cross-Modal & Evidential Fusion)
echo.
set /p member_choice="Select [Default: 1]: "
if "%member_choice%"=="" set member_choice=1

if "%member_choice%"=="1" (
    set "TARGET_BRANCH=prism_contribution_part1_tanush"
    set "USER_NAME=Tanush"
    set "COMMIT_MSG=feat(backbone): Implement ViT-B/16 + DoRA neural model, PRNU filter, and Streamlit UI"
)
if "%member_choice%"=="2" (
    set "TARGET_BRANCH=prism_contribution_part2_somil"
    set "USER_NAME=Somil"
    set "COMMIT_MSG=feat(forensics): Implement ELA compression, optical flow kinematics, and 3D illumination agents"
)
if "%member_choice%"=="3" (
    set "TARGET_BRANCH=prism_contribution_part3_kulsum"
    set "USER_NAME=Kulsum"
    set "COMMIT_MSG=feat(reasoning): Implement biological rPPG, LipSync, Dempster-Shafer fusion, and NIST/SWGDE reasoner"
)

echo.
set /p REPO_URL="Enter GitHub Repository URL [Press Enter for default origin]: "
if "%REPO_URL%"=="" (
    for /f "tokens=*" %%i in ('git config --get remote.origin.url') do set REPO_URL=%%i
)

echo.
echo [1/4] Checking out branch %TARGET_BRANCH%...
git checkout -b %TARGET_BRANCH% 2>nul || git checkout %TARGET_BRANCH%

echo [2/4] Setting author signature for %USER_NAME%...
git config user.name "%USER_NAME%"

echo [3/4] Staging files and creating commit...
git add .
git commit -m "%COMMIT_MSG%"

echo [4/4] Pushing branch to GitHub (%TARGET_BRANCH%)...
git push -u origin %TARGET_BRANCH%

echo.
echo =====================================================================
echo   [SUCCESS] Pushed successfully to %TARGET_BRANCH%!
echo   Next step: Open GitHub and click 'Compare and pull request' to main!
echo =====================================================================
pause
