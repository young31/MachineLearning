# Log Anomaly

- 사용자 로그를 가지고 anomaly detection
- train에 없는 새로운 anomaly class도 포함하여 예측하여야 함
- 자연어처리 문제

## 후기

- validation 데이터를 충분히 활용해 주었어야 하는데 제대로 신경쓰지 못한 부분이 아쉬움
- tokenize 과정이 가장 중요할 것으로 생각해서 내 기준으로 진행하였는데 집중 포인트가 달랐음
  - 생소한 과제 + 제대로 집중하지 못한 과제여서 아쉬움

## Approach

- tfidf, count vectorize
  - 기존 연구들에서 은근히 자주 사용하는 접근법이었음
- w2v
  - anomaly log 들이 비슷한 구조를 가지고 있던 것 같아서 해당 파트를 묶어 줄 수 있을 것으로 기대
- transformer
  - 최근 대중적인 attention 기반의 접근



