"""Translation to create Organic Maps .mwm file with 'isoline' hack"""
import logging
import ogr2osm_translation


FEATURE_ISOLINE_TAG = 'isoline'
ISOLINE_STEPS = 'step_1000', 'step_500', 'step_100', 'step_50', 'step_10', 'zero'

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
                tags[FEATURE_ISOLINE_TAG] = ISOLINE_STEPS[min(max(lvl - 10, 0), len(ISOLINE_STEPS)-1)]
                logging.debug(f'Added zoom-level tag [{FEATURE_ISOLINE_TAG}={tags[FEATURE_ISOLINE_TAG]}], '
                              f'based on [description={desc}]')

        return super().filter_tags(tags)
