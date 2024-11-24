from autoprompt.create_trigger_user import run_model
import logging



# test.py에서 로그 보고 싶을때
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

if __name__ == "__main__":
    user_prompt = "How do I improve my public speaking skills?, I think you are goot at a public presentation"
    personality = [1, 0, 1, 0, 1]  # 사용자 성격 조합

    result = run_model(user_prompt, personality, ckpt_dir="ckpt/")
    print("최종 프롬프트:", result["final_prompt"])
    print("최적 트리거 토큰:", result["best_trigger_tokens"])
    print("최적 성능 메트릭:", result["best_dev_metric"])
