from data_juicer.tools.plan_flow.operator_catalog_service import search

def test_composite_face_requirement_keeps_both_measured_behaviors():
    row=search(['detect number of faces and face bounding box area ratio in image'],modality='image')['results'][0]
    assert row['operator_names'][:2]==['image_face_count_filter','image_face_ratio_filter']
    assert row['ranking'][0]['method']=='measured_behavior_contract'

def test_detection_queries_do_not_offer_destructive_blur_or_removal():
    found=search(['detect image clarity sharpness blur','detect watermark probability in image'],modality='image')
    assert not any('_blur_mapper' in o['name'] or '_remove_mapper' in o['name'] for o in found['operators'])
    exact=search(['image_blur_mapper'],modality='image')
    assert exact['results'][0]['operator_names'][0]=='image_blur_mapper'

def test_multimodal_manifest_does_not_hide_image_scorers():
    found=search(['count faces and face area ratio','nsfw score','watermark probability','aesthetic score'],modality='multimodal')
    names={o['name'] for o in found['operators']}
    assert {'image_face_count_filter','image_face_ratio_filter','image_nsfw_filter','image_watermark_filter','image_aesthetics_filter'} <= names
