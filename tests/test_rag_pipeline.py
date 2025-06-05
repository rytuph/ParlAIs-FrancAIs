# tests/test_rag_pipeline.py
import unittest
import json
import os
from src.rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary user profile file for testing."""
        self.test_profiles_path = 'tests/test_user_profiles.json'
        self.test_profiles = {
            "test_user_1": {
                "error_counts": {"Contractions": 5, "Pronouns": 2}
            }
        }
        with open(self.test_profiles_path, 'w') as f:
            json.dump(self.test_profiles, f)

    def tearDown(self):
        """Remove the temporary file after tests are run."""
        os.remove(self.test_profiles_path)

    def test_query_user_profile_existing_user(self):
        """
        Test that the pipeline correctly retrieves and summarizes
        the profile of an existing user.
        """
        # We only need to test the user profile part, so we can mock the other paths
        pipeline = RAGPipeline(
            vector_db_path='data/vector_store', 
            user_profile_db_path=self.test_profiles_path,
            knowledge_base_path='data/grammar_knowledge_base.json'
        )
        
        context = pipeline._query_user_profile("test_user_1")
        self.assertIn("frequently struggles with 'Contractions'", context)
        self.assertIn("(logged 5 times)", context)

    def test_query_user_profile_new_user(self):
        """
        Test that the pipeline returns a neutral message for a user
        not in the database.
        """
        pipeline = RAGPipeline(
            vector_db_path='data/vector_store', 
            user_profile_db_path=self.test_profiles_path,
            knowledge_base_path='data/grammar_knowledge_base.json'
        )
        
        context = pipeline._query_user_profile("new_user")
        self.assertEqual("Retrieved User Profile: No specific weaknesses logged for this user.", context)

if __name__ == '__main__':
    unittest.main()