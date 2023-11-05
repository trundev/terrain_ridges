# Extra settings
OUR_PATH=$(dirname $_)
MAPS_GENERATOR_OPTS="--without_countries=World --skip=Coastline"

# Check prerequisites
OMIM_PATH=$(readlink -f $OUR_PATH/organicmaps)
if ! [ -d $OMIM_PATH/tools/python ]; then echo "Must clone organicmaps under $OUR_PATH"; return 255&>/dev/null; exit 255; fi
if ! [ -f $OUR_PATH/omim-build-release/generator_tool ]; then echo "Must build organicmaps generator_tool"; return 255&>/dev/null; exit 255; fi
if ! python3 -m ogr2osm --version; then echo "Must install python ogr2osm and GDAL"; return 255&>/dev/null; exit 255; fi

if [ -z "$1" ]; then echo "Need source OGR/OSM"; return 1&>/dev/null; exit 1; fi
ogr=$1


# Convert to OSM if not already OSM/PBF
if [ -n "${ogr/*.osm/}" ] && [ -n "${ogr/*.osm.pbf/}" ]; then
  osm=$ogr.osm
  echo
  echo "#"
  echo "# Convert to OSM"
  echo "#"
  rm $osm
  ogr2osm $ogr -o $osm --positive-id -t $OUR_PATH/ogr2osm_translation-hack.py
else
  osm=$ogr
fi

osm=$(readlink -f $osm)
echo "OSM file: $osm"
if ! md5sum $osm > $osm.md5; then echo "Failed to generate MD5 sum"; return 2&>/dev/null; exit 2; fi

pushd $OMIM_PATH/tools/python

echo
echo "#"
echo "# Update Organic Maps configuration file"
echo "#"
python3 <<HERE-DOC
in_file = 'maps_generator/var/etc/map_generator.ini.default'
out_file = 'maps_generator/var/etc/map_generator.ini'

import configparser
print(f'Reading configuration from "{in_file}"')
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.optionxform = str    # Case-sensitive, preserve case
config.read(in_file)

config["Developer"]["OMIM_PATH"] = "$OMIM_PATH"
config["External"]["PLANET_URL"] = "file://$osm"
print(f'  Developer:OMIM_PATH={config["Developer"]["OMIM_PATH"]}')
print(f'  External:PLANET_URL={config["External"]["PLANET_URL"]}')

print(f'Writting configuration to "{out_file}"')
with open(out_file, 'w') as configfile:
  config.write(configfile)
HERE-DOC

echo
echo "#"
echo "# Generate MWM, options: $MAPS_GENERATOR_OPTS"
echo "#"
python3 -m maps_generator $MAPS_GENERATOR_OPTS

popd
