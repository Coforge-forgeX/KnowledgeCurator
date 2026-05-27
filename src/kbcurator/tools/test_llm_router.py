"""
LLM Router Test Suite

This test file validates the LLM Router setup and functionality.
Run this file to ensure everything is working correctly before using in production.

Usage:
    python test_llm_router.py

The test will:
1. Check environment variables
2. Test provider configuration
3. Validate LLM generation
4. Test provider switching
5. Verify error handling
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, List

# Import from common_adapters (the standard way for all agents)
from common_adapters.configurableAI import ConfigurableAIManager, get_ai_manager

# Import MCP tools for testing (only available in KBCurator)
try:
    from kbcurator.tools.llm_router_tool import (
        use_llm_provider,
        query_llm_router_status, 
        list_llm_providers,
        test_llm_generation,
        reset_llm_router,
        check_default_azure_config
    )
    MCP_TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: MCP tools not available. Testing only ConfigurableAI manager.")
    MCP_TOOLS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMRouterTester:
    """Test suite for LLM Router functionality"""
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.providers_to_test = ["azure", "openai", "gcp", "quasar"]
        
    def log_test_result(self, test_name: str, passed: bool, message: str, details: Dict[str, Any] = None):
        """Log test result"""
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        
        if details:
            for key, value in details.items():
                logger.info(f"    {key}: {value}")
    
    def check_environment_variables(self):
        """Test 1: Check environment variables for all providers"""
        logger.info("=" * 60)
        logger.info("TEST 1: Checking Environment Variables")
        logger.info("=" * 60)
        
        env_checks = {
            "azure": {
                "AZURE_OPENAI_LLM_MODEL_API_KEY": os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY"),
                "AZURE_OPENAI_LLM_MODEL_API_BASE": os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE"),
                "AZURE_OPENAI_LLM_MODEL_LLM_MODEL": os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL"),
                "AZURE_OPENAI_LLM_MODEL_API_VERSION": os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION")
            },
            "openai": {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "OPENAI_ORGANIZATION": os.getenv("OPENAI_ORGANIZATION")
            },
            "gcp": {
                "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
                "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            },
            "quasar": {
                "QUASAR_ENDPOINT_URL": os.getenv("QUASAR_ENDPOINT_URL"),
                "QUASAR_API_KEY": os.getenv("QUASAR_API_KEY"),
                "QUASAR_MODEL": os.getenv("QUASAR_MODEL")
            }
        }
        
        available_providers = []
        
        for provider, vars_dict in env_checks.items():
            required_vars = list(vars_dict.keys())
            if provider == "openai":
                required_vars = ["OPENAI_API_KEY"]  # OPENAI_ORGANIZATION is optional
            elif provider == "quasar":
                required_vars = ["QUASAR_ENDPOINT_URL", "QUASAR_API_KEY"]  # QUASAR_MODEL is optional
            
            missing_vars = [var for var in required_vars if not vars_dict[var]]
            
            if missing_vars:
                self.log_test_result(
                    f"Environment Check - {provider.upper()}",
                    False,
                    f"Missing required environment variables: {', '.join(missing_vars)}",
                    {"missing_vars": missing_vars, "all_vars": vars_dict}
                )
            else:
                available_providers.append(provider)
                self.log_test_result(
                    f"Environment Check - {provider.upper()}",
                    True,
                    "All required environment variables are set",
                    {"vars_status": {k: "SET" if v else "NOT SET" for k, v in vars_dict.items()}}
                )
        
        return available_providers
    
    def test_router_status(self):
        """Test 2: Test router status queries"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Router Status Queries")
        logger.info("=" * 60)
        
        if not MCP_TOOLS_AVAILABLE:
            self.log_test_result(
                "Router Status Query",
                False,
                "MCP tools not available - testing ConfigurableAI manager directly",
                {"note": "This is normal for non-KBCurator agents"}
            )
            return None, None
        
        try:
            # Test status query
            status = query_llm_router_status()
            self.log_test_result(
                "Router Status Query",
                status.get("status") == "success",
                f"Status query returned: {status.get('status', 'unknown')}",
                {
                    "current_provider": status.get("current_provider"),
                    "configured_providers": status.get("configured_providers", []),
                    "available_providers": status.get("available_providers", [])
                }
            )
            
            # Test provider list
            providers = list_llm_providers()
            self.log_test_result(
                "Provider List Query",
                providers.get("status") == "success",
                f"Provider list query returned: {providers.get('status', 'unknown')}",
                {
                    "available_providers": providers.get("available_providers", []),
                    "configured_providers": providers.get("configured_providers", [])
                }
            )
            
            return status, providers
            
        except Exception as e:
            self.log_test_result(
                "Router Status Query",
                False,
                f"Exception occurred: {str(e)}",
                {"exception_type": type(e).__name__}
            )
            return None, None
    
    def test_provider_configuration(self, available_providers: List[str]):
        """Test 3: Test provider configuration"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Provider Configuration")
        logger.info("=" * 60)
        
        configured_providers = []
        
        for provider in available_providers:
            try:
                logger.info(f"\nTesting {provider.upper()} configuration...")
                
                if MCP_TOOLS_AVAILABLE:
                    # Test using MCP tools (KBCurator)
                    result = use_llm_provider(provider)
                    
                    if result.get("status") == "success":
                        configured_providers.append(provider)
                        self.log_test_result(
                            f"Configure {provider.upper()}",
                            True,
                            f"Successfully configured {provider}",
                            {
                                "action": result.get("action"),
                                "current_provider": result.get("current_provider"),
                                "configuration_method": result.get("configuration_method")
                            }
                        )
                    else:
                        self.log_test_result(
                            f"Configure {provider.upper()}",
                            False,
                            f"Failed to configure {provider}: {result.get('message', 'Unknown error')}",
                            {
                                "error_details": result.get("errors", {}),
                                "required_env_vars": result.get("required_env_vars", [])
                            }
                        )
                else:
                    # Test using ConfigurableAI manager directly (other agents)
                    manager = ConfigurableAIManager()
                    manager.configure_from_env(provider)
                    
                    if manager.get_current_provider() == provider:
                        configured_providers.append(provider)
                        self.log_test_result(
                            f"Configure {provider.upper()}",
                            True,
                            f"Successfully configured {provider} via ConfigurableAI",
                            {
                                "current_provider": manager.get_current_provider(),
                                "configured_providers": manager.list_configured_providers()
                            }
                        )
                    else:
                        self.log_test_result(
                            f"Configure {provider.upper()}",
                            False,
                            f"Failed to configure {provider} via ConfigurableAI",
                            {"current_provider": manager.get_current_provider()}
                        )
                    
            except Exception as e:
                self.log_test_result(
                    f"Configure {provider.upper()}",
                    False,
                    f"Exception during {provider} configuration: {str(e)}",
                    {"exception_type": type(e).__name__}
                )
        
        return configured_providers
    
    async def test_llm_generation(self, configured_providers: List[str]):
        """Test 4: Test LLM text generation"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: LLM Text Generation")
        logger.info("=" * 60)
        
        test_prompt = "What is artificial intelligence? Answer in one sentence."
        
        for provider in configured_providers:
            try:
                logger.info(f"\nTesting text generation with {provider.upper()}...")
                
                if MCP_TOOLS_AVAILABLE:
                    # Test using MCP tools (KBCurator)
                    # Switch to provider first
                    switch_result = use_llm_provider(provider)
                    if switch_result.get("status") != "success":
                        self.log_test_result(
                            f"Generate Text - {provider.upper()}",
                            False,
                            f"Failed to switch to {provider}",
                            {"switch_result": switch_result}
                        )
                        continue
                    
                    # Test generation
                    result = await test_llm_generation(
                        prompt=test_prompt,
                        provider=provider,
                        max_tokens=100,
                        temperature=0.7
                    )
                    
                    if result.get("status") == "success":
                        response = result.get("response", "")
                        self.log_test_result(
                            f"Generate Text - {provider.upper()}",
                            True,
                            f"Successfully generated text ({len(response)} chars)",
                            {
                                "provider": result.get("provider"),
                                "response_preview": response[:100] + "..." if len(response) > 100 else response,
                                "response_time": result.get("metadata", {}).get("response_time_seconds"),
                                "response_length": result.get("metadata", {}).get("response_length")
                            }
                        )
                    else:
                        self.log_test_result(
                            f"Generate Text - {provider.upper()}",
                            False,
                            f"Failed to generate text: {result.get('message', 'Unknown error')}",
                            {"error_details": result}
                        )
                else:
                    # Test using ConfigurableAI manager directly (other agents)
                    manager = ConfigurableAIManager()
                    manager.configure_from_env(provider)
                    
                    import time
                    start_time = time.time()
                    response = await manager.generate_text(
                        test_prompt,
                        max_tokens=100,
                        temperature=0.7
                    )
                    end_time = time.time()
                    
                    self.log_test_result(
                        f"Generate Text - {provider.upper()}",
                        True,
                        f"Successfully generated text ({len(response)} chars)",
                        {
                            "provider": provider,
                            "response_preview": response[:100] + "..." if len(response) > 100 else response,
                            "response_time": round(end_time - start_time, 3),
                            "response_length": len(response)
                        }
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Generate Text - {provider.upper()}",
                    False,
                    f"Exception during text generation: {str(e)}",
                    {"exception_type": type(e).__name__}
                )
    
    def test_provider_switching(self, configured_providers: List[str]):
        """Test 5: Test provider switching"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: Provider Switching")
        logger.info("=" * 60)
        
        if len(configured_providers) < 2:
            self.log_test_result(
                "Provider Switching",
                False,
                f"Need at least 2 configured providers for switching test. Found: {len(configured_providers)}",
                {"configured_providers": configured_providers}
            )
            return
        
        # Test switching between first two providers
        provider1, provider2 = configured_providers[0], configured_providers[1]
        
        try:
            # Switch to first provider
            result1 = use_llm_provider(provider1)
            status1 = query_llm_router_status()
            
            # Switch to second provider
            result2 = use_llm_provider(provider2)
            status2 = query_llm_router_status()
            
            # Switch back to first provider
            result3 = use_llm_provider(provider1)
            status3 = query_llm_router_status()
            
            success = (
                result1.get("status") == "success" and
                result2.get("status") == "success" and
                result3.get("status") == "success" and
                status1.get("current_provider") == provider1 and
                status2.get("current_provider") == provider2 and
                status3.get("current_provider") == provider1
            )
            
            self.log_test_result(
                "Provider Switching",
                success,
                f"Successfully switched between {provider1} and {provider2}",
                {
                    "switch_sequence": [provider1, provider2, provider1],
                    "final_provider": status3.get("current_provider"),
                    "switch_results": [
                        result1.get("action"),
                        result2.get("action"), 
                        result3.get("action")
                    ]
                }
            )
            
        except Exception as e:
            self.log_test_result(
                "Provider Switching",
                False,
                f"Exception during provider switching: {str(e)}",
                {"exception_type": type(e).__name__}
            )
    
    def test_error_handling(self):
        """Test 6: Test error handling"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 6: Error Handling")
        logger.info("=" * 60)
        
        # Test invalid provider
        try:
            result = use_llm_provider("invalid_provider")
            expected_error = result.get("status") == "error"
            
            self.log_test_result(
                "Invalid Provider Error",
                expected_error,
                f"Invalid provider correctly returned error status: {expected_error}",
                {
                    "result_status": result.get("status"),
                    "error_message": result.get("message"),
                    "error_type": result.get("error_type")
                }
            )
            
        except Exception as e:
            self.log_test_result(
                "Invalid Provider Error",
                True,  # Exception is expected
                f"Invalid provider correctly raised exception: {str(e)}",
                {"exception_type": type(e).__name__}
            )
    
    def test_azure_specific(self):
        """Test 7: Azure-specific functionality"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 7: Azure-Specific Tests")
        logger.info("=" * 60)
        
        try:
            azure_status = check_default_azure_config()
            
            self.log_test_result(
                "Azure Default Config Check",
                azure_status.get("status") == "success",
                f"Azure config check completed with status: {azure_status.get('status')}",
                {
                    "azure_configured": azure_status.get("azure_configured"),
                    "azure_is_current": azure_status.get("azure_is_current_provider"),
                    "all_env_vars_present": azure_status.get("all_azure_env_vars_present"),
                    "default_setup_successful": azure_status.get("default_setup_successful")
                }
            )
            
        except Exception as e:
            self.log_test_result(
                "Azure Default Config Check",
                False,
                f"Exception during Azure config check: {str(e)}",
                {"exception_type": type(e).__name__}
            )
    
    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("Starting LLM Router Test Suite")
        logger.info("=" * 80)
        
        # Reset router before testing (only if MCP tools available)
        if MCP_TOOLS_AVAILABLE:
            try:
                reset_result = reset_llm_router()
                logger.info(f"Router reset: {reset_result.get('status', 'unknown')}")
            except Exception as e:
                logger.warning(f"Failed to reset router: {e}")
        else:
            logger.info("Testing ConfigurableAI manager directly (no MCP reset needed)")
        
        # Run tests
        available_providers = self.check_environment_variables()
        status, providers = self.test_router_status()
        configured_providers = self.test_provider_configuration(available_providers)
        
        if configured_providers:
            await self.test_llm_generation(configured_providers)
            self.test_provider_switching(configured_providers)
        else:
            logger.warning("No providers configured - skipping generation and switching tests")
        
        self.test_error_handling()
        self.test_azure_specific()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = [r for r in self.test_results if r["passed"]]
        failed_tests = [r for r in self.test_results if not r["passed"]]
        
        logger.info(f"Total Tests: {len(self.test_results)}")
        logger.info(f"Passed: {len(passed_tests)}")
        logger.info(f"Failed: {len(failed_tests)}")
        logger.info(f"Success Rate: {len(passed_tests)/len(self.test_results)*100:.1f}%")
        
        if failed_tests:
            logger.info("\n❌ FAILED TESTS:")
            for test in failed_tests:
                logger.info(f"  - {test['test']}: {test['message']}")
        
        if passed_tests:
            logger.info("\n✅ PASSED TESTS:")
            for test in passed_tests:
                logger.info(f"  - {test['test']}: {test['message']}")
        
        # Recommendations
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 80)
        
        env_failed = [r for r in failed_tests if "Environment Check" in r["test"]]
        if env_failed:
            logger.info("Set up environment variables for failed providers:")
            for test in env_failed:
                provider = test["test"].split(" - ")[1].lower()
                missing_vars = test["details"].get("missing_vars", [])
                logger.info(f"  {provider}: {', '.join(missing_vars)}")
        
        config_failed = [r for r in failed_tests if "Configure" in r["test"]]
        if config_failed:
            logger.info("Configuration issues found - check API keys and endpoints")
        
        generation_failed = [r for r in failed_tests if "Generate Text" in r["test"]]
        if generation_failed:
            logger.info("Text generation issues - verify network connectivity and API limits")
        
        if not failed_tests:
            logger.info("All tests passed! LLM Router is ready for production use.")
        
        logger.info("\n" + "=" * 80)


def main():
    """Main test execution"""
    print("LLM Router Test Suite")
    print("====================")
    print("This will test your LLM Router configuration and functionality.")
    print("Make sure you have set the appropriate environment variables.\n")
    
    # Run tests
    tester = LLMRouterTester()
    asyncio.run(tester.run_all_tests())
    
    print("\nTest suite completed!")
    print("Check the output above for any issues that need to be resolved.")
    print("Refer to LLM_ROUTER_DOCUMENTATION.md for configuration help.")


if __name__ == "__main__":
    main()