import os
import math
import logging
import ogr2osm

# Default spheroid radius
DEF_SPHEROID_RADIUS = 6378137

def get_tag_val(desc, n):
    """Extract specific value from 'description', expected format '<name>: <val> km,'"""
    _, *v = desc.split(n + ':')
    if not v:
        return None
    # Drop the rest of string
    v, *_ = v[0].split(',')
    # Drop the suffix (like km, km2)
    return v.strip().split()[0]

class obf_translation(ogr2osm.TranslationBase):
    spheroid_radius = DEF_SPHEROID_RADIUS

    def __init__(self):
        self.maxzoom = os.environ.get('MAXZOOM')
        if self.maxzoom is not None:
            self.maxzoom = float(self.maxzoom)
            logging.info(f'obf_translation: MAXZOOM {self.maxzoom}')

    def get_zoomlevel(self, desc):
        ar = get_tag_val(desc, 'area')
        if ar is None:
            return None
        ar = float(ar) * 1e6    # Convert to square meters
        lvl0_area = 4 * math.pi * self.spheroid_radius**2
        return math.log2(lvl0_area / ar) / 2

    def filter_layer(self, layer):
        radius = layer.GetSpatialRef().GetAttrValue('SPHEROID', 1)
        if radius is not None:
            self.spheroid_radius = float(radius)
        logging.debug(f'obf_translation / filter_layer: Name="{layer.GetName()}", radius={self.spheroid_radius}')
        return layer

    def filter_feature(self, ogrfeature, layer_fields, reproject):
        if self.maxzoom is None:
            return ogrfeature
        desc = ogrfeature.GetField('description')
        if desc is None:
            return ogrfeature
        lvl = self.get_zoomlevel(desc)
        if lvl is None:
            return ogrfeature

        # Filter out too detailed features
        return None if round(self.maxzoom - lvl) < 0 else ogrfeature

    def filter_tags(self, tags):
        """Break-down 'description' tag into 'length' and 'surface'"""
        desc = tags.get('description')
        if desc is not None:
            # In OsmAnd, "distance" refers to the path-length, while "length"/"width" -- to the bounding extents
            # See: https://wiki.openstreetmap.org/wiki/Key:distance, https://wiki.openstreetmap.org/wiki/Key:length
            ln = get_tag_val(desc, 'length')
            if ln is not None:
                tags['distance'] = ln
            ar = get_tag_val(desc, 'area')
            if ar is not None:
                tags['surface'] = ar
            ct = None
            lvl = self.get_zoomlevel(desc)
            if lvl and self.maxzoom:
                idx = round(self.maxzoom - lvl)
                # Snap to '100m' for idx above 3
                ct = (['10m', '20m', '50m'][idx:] or ['100m'])[0]
                tags['contourtype'] = ct
            del tags['description']
            logging.debug(f'Converting "{desc}" (zoom-level={lvl:.1f}) to distance="{ln}", surface="{ar}", contourtype="{ct}')
        return tags
