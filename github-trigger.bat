set REPO=%1
if _%REPO% == _ set REPO=terrain_ridges

set WORKFLOW=main.yml
set WORKFLOW_REF=master
set WORKFLOW_INPUTS=\"SRC_URL\": \"https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/Eurasia/N39E021.hgt.zip\"
set WORKFLOW_INPUTS=%WORKFLOW_INPUTS%, \"DST_EXT\": \".kmz\"
set WORKFLOW_INPUTS=%WORKFLOW_INPUTS%, \"TOOL_OPTIONS\": \"\"
set WORKFLOW_INPUTS=%WORKFLOW_INPUTS%, \"GDALWARP_OPTIONS\": \"-ts 600 600\"
set WORKFLOW_INPUTS=%WORKFLOW_INPUTS%, \"REBUILD_CACHE\": \"false\"

set OPT=
set OPT=-H "Accept: application/vnd.github.v3+json" %OPT%
::set OPT=--user trundev:<set GITHUB_TOKEN here> %OPT%

::See https://docs.github.com/en/rest/reference/actions#create-a-workflow-dispatch-event
@echo ============================================
@echo Create a workflow dispatch event (HTTP POST)
curl -X POST %OPT% https://api.github.com/repos/trundev/%REPO%/actions/workflows/%WORKFLOW%/dispatches -d "{\"ref\": \"%WORKFLOW_REF%\", \"inputs\": {%WORKFLOW_INPUTS%}}"
@echo.
