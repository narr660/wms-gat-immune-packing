from src.data_utils_v3 import augment_item, check_relation_v3

def test_deterministic():
    item={'size':[2,3,4],'category':'식품','category_id':1,'hazard_level':0.7}
    a=augment_item(item,dataset_name='rs',seq_idx=0,item_idx=0)
    b=augment_item(item,dataset_name='rs',seq_idx=0,item_idx=0)
    assert a==b

def test_symmetric():
    base={'size':[2,3,4],'category':'식품','category_id':1,'hazard_level':0.7}
    a=augment_item(base,dataset_name='rs',seq_idx=0,item_idx=0)
    b=augment_item(base,dataset_name='rs',seq_idx=0,item_idx=1)
    x=check_relation_v3(a,b); y=check_relation_v3(b,a)
    assert (x['base_risk'],x['context_risk'],x['allowed'])==(y['base_risk'],y['context_risk'],y['allowed'])
