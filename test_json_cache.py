import json

def test():
    # Simulate data
    user_word_sets = {
        "did:plc:123": {"word1", "word2"},
        "did:plc:456": {"word3"}
    }
    all_community_users = ["did:plc:123", "did:plc:456", "did:plc:789"]

    # Dump
    serializable_user_word_sets = {k: list(v) for k, v in user_word_sets.items()}
    with open("test_cache.json", "w", encoding="utf-8") as f:
        json.dump((serializable_user_word_sets, all_community_users), f)

    # Load
    with open("test_cache.json", "r", encoding="utf-8") as f:
        loaded_user_word_sets, loaded_users = json.load(f)
        loaded_user_word_sets = {k: set(v) for k, v in loaded_user_word_sets.items()}

    # Verify
    assert loaded_users == all_community_users
    assert loaded_user_word_sets == user_word_sets
    print("Serialization and deserialization successful!")

test()
