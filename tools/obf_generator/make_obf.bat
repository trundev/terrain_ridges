@echo off
if not exist "%1" echo Missing source OGR file "%1" && exit /b 255
echo Processing "%1"...

::
:: Step 1 breakdown into zoom-levels and simplify
:: Max zoom: L10 - m / pixel (on Equator): 152.746
::
set ZOOM_TOL_LEVELS="10 .004" "11 .002" "12 .001" "13 .0005" "14 .00025"
if exist "%1.tmp" echo Folder "%1.tmp" will be removed! Press Ctrl-C to avoid this && pause && rmdir "%1.tmp" /s/q
mkdir "%1.tmp"
echo Zoom levels / simplify tolerance: %ZOOM_TOL_LEVELS%
echo. && echo === && echo.

for %%x in (%ZOOM_TOL_LEVELS%) do for /f "tokens=1,2" %%g in (%%x) do (
    echo Simplifying zoom %%g, tolerance %%h
    python -m osgeo_utils.samples.ogr2ogr -f KML "%1.tmp\%%g.kml" "%1" -simplify %%h
    if errorlevel 1 echo Error: GDAL ogr2ogr failed && exit /b 1
)
echo. && echo === && echo.

::
:: Step 2 convert individual zoom-levels to OSM files
:: Needs ogr2osm: "pip install ogr2osm"
::  tested with "pip install git+https://github.com/roelderickx/ogr2osm@v1.1.1"
::
goto :skip
:: Create updated batch.xml: process_attributes.mapZooms
::  %1 - source batch.xml
::  %2 - target batch.xml
::  %3 - New value for process_attributes.mapZooms
:process_batch_xml
powershell -command "&{$xml=[xml](gc %1); $nsm = New-Object Xml.XmlNamespaceManager($xml.NameTable); $nsm.AddNamespace('ns', $xml.DocumentElement.NamespaceURI); $node=$xml.SelectSingleNode('//ns:process_attributes', $nsm); $node.mapZooms='%3'; $xml.Save('%2')}"
goto :eof
:skip

echo Processing "%1.tmp" zoom-levels...
pushd "%1.tmp"
set LAST_ZOOM=
for %%x in (%ZOOM_TOL_LEVELS%) do for /f %%x in (%%x) do (
    echo Zoom: %%x to L_%%x.osm
    set MAXZOOM=%%x
    python -m ogr2osm "%%x.kml" -o L_%%x.osm -t %~dp0ogr2osm_translation.py
    if errorlevel 1 echo Error: ogr2osm failed && popd && exit /b 2
    :: Create batch.xml-s (L_<z>.osm.xml) with correct process_attributes.mapZooms
    call :process_batch_xml %~dp0batch.xml L_%%x.osm.xml %%x
    set LAST_ZOOM=%%x
)
:: The batch.xml for the last zoom must end with "-"
call :process_batch_xml %~dp0batch.xml L_%LAST_ZOOM%.osm.xml %LAST_ZOOM%-
popd
if errorlevel 1 exit /b %errorlevel%
echo. && echo === && echo.

::
:: Step 3 convert the OSM files to OBF
:: Needs OsmAndMapCreator from http://download.osmand.net/latest-night-build/OsmAndMapCreator-main.zip
:: See https://github.com/osmandapp/OsmAnd-tools/tree/master/java-tools/OsmAndMapCreator
::
set JAVA="%JAVA_HOME%\bin\java.exe"
if not exist %JAVA% echo Missing JAVA, please install and set JAVA_HOME && exit /b 255
set OAMC_JAVA=%JAVA% -Djava.util.logging.config.file=logging.properties -Xms64M -Xmx6300M
set OAMC_JAVA=%OAMC_JAVA% -cp "%~dp0OsmAndMapCreator/OsmAndMapCreator.jar;%~dp0OsmAndMapCreator/lib/*.jar"

echo Converting OSM files...
pushd "%1.tmp"
mkdir osm
for %%x in (L_*.osm) do (
    del osm\*.* /q
    copy %%x osm
    :: Use already created batch.xml-s (L_<z>.osm.xml)
    %OAMC_JAVA% net.osmand.util.IndexBatchCreator %%x.xml
    if errorlevel 1 echo Error: OsmAndMapCreator / IndexBatchCreator failed && popd && exit /b 3
)
:: Run OBF inspector
for %%x in (L_*.obf) do (
    echo. && echo --- %%x ---
    %OAMC_JAVA% net.osmand.obf.BinaryInspector -v %%x
    if errorlevel 1 echo Error: OsmAndMapCreator / BinaryInspector failed && popd && exit /b 4
)
popd
if errorlevel 1 exit /b %errorlevel%
echo. && echo === && echo.

::
:: Step 4 merge generated OBFs
::
echo Merging final OBF (%~dpn1.obf)...
del %~dpn1.obf /q
%OAMC_JAVA% net.osmand.MainUtilities merge-flat-obf "%~dpn1.obf" %1.tmp\*.obf
echo. && echo --- %~dpn1.obf ---
%OAMC_JAVA% net.osmand.obf.BinaryInspector -v %~dpn1.obf
