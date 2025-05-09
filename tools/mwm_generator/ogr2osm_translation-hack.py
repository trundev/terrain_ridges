"""Translation to create Organic Maps .mwm file with 'isoline' hack"""
import logging
import ogr2osm_translation


FEATURE_RIDGE_MARKER = {}
FEATURE_ZOOM_LEVEL_TAG = 'highway'
# Map from ridge zoom-level to tag value (starting from level 8)
FEATURE_ZOOM_LEVEL_VALS = ['primary', 'secondary', 'tertiary', 'tertiary', 'road', 'road', 'footway', 'footway']

class mwm_translation(ogr2osm_translation.mwm_translation):
    """Hack ogr2osm translation"""

    def filter_tags(self, tags):
        """Add 'isoline' tags to make fake contour-lines (avoid filtering-out)"""
        # Note: Run this before `super().filter_tags` as it will remove `description` tag
        desc = tags.get('description', tags.get('Description'))
        if desc is not None:
            lvl = self._get_zoomlevel(desc)
            if lvl:
                lvl = round(lvl)
                tags |= FEATURE_RIDGE_MARKER
                tags[FEATURE_ZOOM_LEVEL_TAG] = FEATURE_ZOOM_LEVEL_VALS[
                        min(max(lvl - 8, 0), len(FEATURE_ZOOM_LEVEL_VALS)-1)]
                logging.debug('Added zoom-level tag '
                              f'[{FEATURE_ZOOM_LEVEL_TAG}={tags[FEATURE_ZOOM_LEVEL_TAG]}], '
                              f'based on [description={desc}]')

        return super().filter_tags(tags)
