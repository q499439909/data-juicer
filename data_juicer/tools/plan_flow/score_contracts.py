"""Curated score semantics; unknown contracts remain explicitly unknown."""
SCORES = {
    'image_face_count_filter': ('face_counts', [0, None], 'count', 'Detected faces, not face clarity or identity.'),
    'image_face_ratio_filter': ('face_ratios', [0, 1], 'range', 'Largest detected face rectangle area / image area. No face produces 0. Does not measure clarity.'),
    'image_nsfw_filter': ('image_nsfw_score', [0, 1], 'lower', 'Model NSFW score; not a calibrated guarantee of safety.'),
    'image_watermark_filter': ('image_watermark_prob', [0, 1], 'lower', 'Model watermark probability; threshold needs domain validation.'),
    'image_aesthetics_filter': ('image_aesthetics_scores', [None, None], 'higher', 'Model-specific aesthetics score; not an objective quality or face-clarity measurement.'),
}

OUTPUT_CONTRACTS = {
    'image_tagging_vlm_mapper': {
        'kind': 'model_generated_tags',
        'model_response_schema': {
            'type': 'object',
            'required': ['tags'],
            'additionalProperties': False,
            'properties': {
                'tags': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'minItems': 1,
                    'maxItems': 10,
                },
            },
        },
        'canonical_response_field': 'tags',
        'tolerated_response_fields': ['tag'],
        'canonical_example': {'tags': ['tag1', 'tag2']},
        'ordering_guaranteed': False,
        'normalization': [
            'lowercase',
            'spaces_to_hyphens',
            'truncate_each_tag_to_30_characters',
            'deduplicate',
            'keep_at_most_10_tags',
        ],
        'storage_path': '__dj__meta__.<tag_field_name>',
        'storage_shape': 'array of tag arrays, one array per input image in original image order',
        'empty_tags_behavior': 'API mode retries try_num times, then fails the Run',
        'prompt_parameter': 'system_prompt',
    },
}

# Search these measured behaviors before broad lexical similarity. Composite
# requests can match multiple contracts; this does not imply task-quality proof.
SEARCH_BEHAVIORS = {
    'image_face_count_filter': (('face','人脸'),('count','number','exactly','数量','恰好','几张')),
    'image_face_ratio_filter': (('face','人脸'),('ratio','area','比例','面积','占比')),
    'image_nsfw_filter': (('nsfw','adult content','色情'),),
    'image_watermark_filter': (('watermark','水印'),),
    'image_aesthetics_filter': (('aesthetic','美学'),),
}

def search_behaviors(query):
    q=query.casefold()
    return [name for name,groups in SEARCH_BEHAVIORS.items() if all(any(term in q for term in alternatives) for alternatives in groups)]


def enrich(item):
    score=SCORES.get(item['name'])
    limitations=list(item.get('validation_summary',{}).get('limitations',[]))
    output_contract = OUTPUT_CONTRACTS.get(item['name'], {'status': 'unknown'})
    result={**item,'input_contract':{'kind':'DJ record','image_key':'images (configurable)','image_values':'array of local media references'} if score else {'kind':'DJ record','status':'consult_parameters'},
            'output_contract':output_contract,'score_semantics':[],
            'runtime_requirements':{'model_locks':item.get('model_locks',[]),'availability':'prepare_plan runtime_assessment'},
            'limitations':limitations}
    if score:
        field,bounds,direction,note=score
        result['output_contract']={'stats_field':'__dj__stats__.'+field,'type':'array[number]','alignment':'one scalar per input image, original image order',
                                   'preserve_rejected':'Use native image_audit; it invokes Filter.run(reduce=False) then applies explicit decision rules.'}
        result['score_semantics']=[{'field':field,'range':bounds,'preferred_direction':direction,'meaning':note}]
        result['limitations'].append(note)
    elif item['name'] not in OUTPUT_CONTRACTS:
        result['limitations'].append('Output and score contract is not curated for this operator. Do not invent field names or quality guarantees; use bounded diagnosis or validated custom contract.')
    return result
