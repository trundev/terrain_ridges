"""Translation to create Organic Maps .mwm file

See https://github.com/roelderickx/ogr2osm#translations
"""
import os
import math
import logging
import ogr2osm

# Tag to store the feature extent/zoom-level (integer)
FEATURE_ZOOM_LEVEL_TAG = 'size'

# Environment variable to reduce the over-detailed features
MAXZOOM_ENV_VAR = 'MAXZOOM'

def get_tag_val(desc, n):
    """Extract specific value from 'description', expected format '<name>: <val> km,'"""
    _, *v = desc.split(n + ':')
    if not v:
        return None
    # Drop the rest of string
    v, *_ = v[0].split(',')
    # Drop the suffix (like km, km2)
    return v.strip().split()[0]

class mwm_translation(ogr2osm.TranslationBase):
    """ogr2osm translation"""
    spheroid_radius = 6378137   # Default earth spheroid radius

    def __init__(self):
        self.maxzoom = os.environ.get(MAXZOOM_ENV_VAR)
        if self.maxzoom is not None:
            self.maxzoom = float(self.maxzoom)
        logging.info(f'Started, MAXZOOM {self.maxzoom}')

    def _get_zoomlevel(self, desc):
        """Calculate zoom-level from 'area' tag (to be as 'ridge_rank')"""
        ar = get_tag_val(desc, 'area')
        if ar is None:
            logging.warning(f'Unable to extract feature-area from description "{desc}"')
            return None
        ar = float(ar) * 1e6    # Convert to square meters
        lvl0_area = 4 * math.pi * self.spheroid_radius**2
        return math.log2(lvl0_area / ar) / 2

    def filter_layer(self, layer):
        """Filter layers"""
        radius = layer.GetSpatialRef().GetAttrValue('SPHEROID', 1)
        if radius is not None:
            self.spheroid_radius = float(radius)
        logging.info(f'Layer "{layer.GetName()}", radius={self.spheroid_radius}')
        return layer

    def filter_feature(self, ogrfeature, layer_fields, reproject):
        """Filter feature"""
        logging.debug(f'filter_feature: name={ogrfeature.GetField("name")}')
        # Filter out too detailed features, if MAXZOOM is set
        if self.maxzoom is None:
            return ogrfeature
        desc = ogrfeature.GetField('description')
        if desc is None:
            return ogrfeature
        lvl = self._get_zoomlevel(desc)
        if lvl is None or self.maxzoom >= lvl:
            return ogrfeature

        logging.debug(f'Discard high-detail feature: {ogrfeature.GetField("name")}, ridge_rank: {lvl:.1f}, max: {self.maxzoom}')
        return None

    def filter_tags(self, tags):
        """Filter feature tags"""
        desc = tags.pop('description', tags.pop('Description', None))
        if desc is not None:
            lvl = self._get_zoomlevel(desc)
            if lvl:
                lvl = round(lvl)
                tags[FEATURE_ZOOM_LEVEL_TAG] = str(lvl)
                logging.debug(f'Added zoom-level tag [{FEATURE_ZOOM_LEVEL_TAG}={lvl}], '
                              f'based on [description={desc}]')
            # Ensure [natural=ridge] is present, as it can be dropped by some OGR formats/drivers
            if 'natural' not in tags:
                logging.warning(f'Feature tags "{tags}" do NOT include "natural" key. Using value "ridge"')
                tags['natural'] = 'ridge'
            # 'name' tag is the feature label, if not available take 'Name' value
            if not tags.get('name'):
                tags['name'] = tags.get('Name')
        return tags

# Use same log-level as ogr2osm, or INFO
#logging.getLogger().setLevel(logging.getLogger('ogr2osm').level)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().name = __name__
