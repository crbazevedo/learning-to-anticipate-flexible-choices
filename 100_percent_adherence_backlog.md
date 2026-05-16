# Backlog Técnico: 100% Aderência à Tese

> **Canon note (2026-05-16, W1-5):** This file is written in Portuguese
> as a working backlog. The canonical algorithmic spec for the open-
> source release is the IEEE Transactions on Cybernetics paper
> (`docs/paper.pdf`; Azevedo & Von Zuben, 2015), which uses its own
> equation numbering. Every body citation of a thesis equation has been
> annotated inline with its paper-equation counterpart. For the full
> reconciliation table see §0 of `thesis_codebase_analysis.md`. Full
> English translation is queued as a separate later unit; this unit
> only adds paper-canonical anchoring.

## Status Atual: 85% → Meta: 100%

### Resumo Executivo
O codebase atual possui **85% de aderência** à tese (revisão baseada em análise mais detalhada). Para alcançar **100%**, precisamos implementar **4 componentes críticos** e **3 melhorias avançadas**. O backlog está organizado por prioridade e complexidade técnica.

### Principais Descobertas da Segunda Análise
- **✅ TIP Calculation**: Já implementado em múltiplos experimentos, precisa apenas de integração
- **✅ Constraint Handling**: Bem implementado com ε-feasibility
- **✅ Experimental Setup**: Parâmetros configuráveis, alguns diferentes da tese
- **⚠️ Sliding Window Dirichlet**: Principal gap restante

---

## 🎯 **PRIORIDADE 1: Componentes Críticos (15% restante)**

### **EPIC 1: Sliding Window Dirichlet Model**
**Impacto**: Alto | **Complexidade**: Média | **Estimativa**: 2-3 sprints

#### **User Story 1.1: Implementar Equações 6.24-6.27**
```python
# Arquivo: src/algorithms/sliding_window_dirichlet.py
class SlidingWindowDirichlet:
    def __init__(self, window_size_K: int, concentration_scaling_s: float):
        self.K = window_size_K
        self.s = concentration_scaling_s
        self.alpha_history = []
        self.alpha_0 = None  # Initial concentration parameter
    
    def update_concentration(self, t: int, u_t_minus_1: np.ndarray) -> np.ndarray:
        """
        Implement Equations 6.24-6.27 / paper Eqs (17)-(18) for concentration parameter updates
        
        Equation 6.24 / paper Eq (17), case t<K: α_t^(i) | u_{t-1}^(i) = α_{t-1}^(i) | u_{t-2}^(i) + s u_{t-1}^(i) if t < K
        Equation 6.25 / paper Eq (17), case t=K: α_t^(i) | u_{t-1}^(i) = α_{t-1}^(i) | u_{t-2}^(i) + s u_{t-1}^(i) - α_0^(i) if t = K
        Equation 6.26 / paper Eq (17), case t>K: α_t^(i) | u_{t-1}^(i) = α_{t-1}^(i) | u_{t-2}^(i) + s u_{t-1}^(i) - s u_{t-K-1}^(i) if t > K
        """
        if t == 0:
            # Initialize with even-handed concentration
            alpha_t = self.s * np.ones_like(u_t_minus_1) / len(u_t_minus_1)
            self.alpha_0 = alpha_t.copy()
        elif t < self.K:
            # Equation 6.24 / paper Eq (17), case t<K: Accumulating observations
            alpha_t = self.alpha_history[-1] + self.s * u_t_minus_1
        elif t == self.K:
            # Equation 6.25 / paper Eq (17), case t=K: First time window is full
            alpha_t = self.alpha_history[-1] + self.s * u_t_minus_1 - self.alpha_0
        else:
            # Equation 6.26 / paper Eq (17), case t>K: Sliding window
            alpha_t = (self.alpha_history[-1] + self.s * u_t_minus_1 - 
                      self.s * self.alpha_history[-(self.K+1)])
        
        self.alpha_history.append(alpha_t)
        return alpha_t
```

**Critérios de Aceitação**:
- [ ] Implementar todas as 3 equações (6.24-6.26)
- [ ] Testes unitários com dados sintéticos
- [ ] Integração com `DirichletPredictor` existente
- [ ] Validação contra implementação C++

#### **User Story 1.2: Implementar Equação 6.28 (Velocity)**
```python
def calculate_velocity(self, t: int) -> np.ndarray:
    """
    Implement Equation 6.28 / paper Eq (19): Velocity calculation
    
    ẋ_t^(i) = s u_{t-1}^(i) - α_{t-K}^(i) | u_{t-K-1}^(i)
    """
    if t < self.K:
        return np.zeros_like(self.alpha_history[-1])
    
    current_alpha = self.alpha_history[-1]
    past_alpha = self.alpha_history[-(self.K+1)]
    
    # Velocity = current - past (simplified)
    velocity = current_alpha - past_alpha
    return velocity
```

**Critérios de Aceitação**:
- [ ] Implementar cálculo de velocidade
- [ ] Integrar com predição Dirichlet
- [ ] Testes de validação

---

### **EPIC 2: Integração TIP com Algoritmo Principal**
**Impacto**: Alto | **Complexidade**: Baixa | **Estimativa**: 1 sprint

#### **User Story 2.1: Integrar TIP Calculation no AnticipatoryLearning**
```python
# Arquivo: src/algorithms/anticipatory_learning.py
class AnticipatoryLearning:
    def __init__(self, ...):
        # Adicionar configurações TIP
        self.tip_calculator = TemporalIncomparabilityCalculator()
        self.binary_entropy_enabled = True
        
    def calculate_tip(self, current_solution: Solution, predicted_solution: Solution) -> float:
        """
        Calculate Temporal Incomparability Probability (Definition 6.1)
        P_{t,t+h} = Pr[ẑ_t || ẑ_{t+h} | ẑ_t]
        """
        # Implementar usando distribuições Gaussianas marginais
        current_roi, current_risk = current_solution.P.ROI, current_solution.P.risk
        predicted_roi, predicted_risk = predicted_solution.P.ROI, predicted_solution.P.risk
        
        # Usar covariância do Kalman Filter
        current_cov = current_solution.P.kalman_state.P[:2, :2]
        predicted_cov = predicted_solution.P.kalman_state.P[:2, :2]
        
        # Calcular probabilidade de não-dominância mútua
        tip = self._calculate_mutual_non_dominance_probability(
            current_roi, current_risk, current_cov,
            predicted_roi, predicted_risk, predicted_cov
        )
        
        return tip
    
    def _binary_entropy(self, p: float) -> float:
        """Calculate binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)"""
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    
    def calculate_anticipatory_learning_rate_tip(self, tip: float, horizon: int) -> float:
        """
        Calculate anticipatory learning rate using TIP (Equation 6.6 / paper Eq (13))
        λ_{t+h} = (1/(H-1)) * [1 - H(p_{t,t+h})]
        """
        entropy = self._binary_entropy(tip)
        return (1.0 / (horizon - 1)) * (1.0 - entropy)
```

**Critérios de Aceitação**:
- [ ] Integrar TIP calculation no `AnticipatoryLearning` class
- [ ] Implementar binary entropy function
- [ ] Implementar Equation 6.6 / paper Eq (13) para learning rate
- [ ] Usar distribuições Gaussianas marginais corretas
- [ ] Testes unitários para TIP calculation

---

### **EPIC 3: Correspondence Mapping Completo**
**Impacto**: Médio | **Complexidade**: Baixa | **Estimativa**: 1 sprint

#### **User Story 3.1: Implementar Mapeamento Explícito**
```python
# Arquivo: src/algorithms/correspondence_mapper.py
class CorrespondenceMapper:
    def __init__(self):
        self.ranked_solutions_history = []
        self.solution_identity_map = {}
        self.rank_persistence = {}
    
    def map_solutions_across_time(self, current_solutions: List[Solution], 
                                 previous_solutions: List[Solution]) -> Dict:
        """
        Implement correspondence mapping for ranked solutions
        
        Based on Section 6.2.2 of thesis:
        - Sort solutions by objective function values
        - Maintain rank correspondence across time periods
        - Track individual solution evolution
        """
        # Sort current solutions by first objective (ROI)
        current_sorted = sorted(current_solutions, key=lambda s: s.P.ROI)
        
        # Create correspondence mapping
        correspondence = {}
        for i, solution in enumerate(current_sorted):
            solution.rank = i
            solution.rank_ROI = i
            
            # Map to previous solution if exists
            if i < len(previous_solutions):
                correspondence[i] = {
                    'current': solution,
                    'previous': previous_solutions[i],
                    'rank_persistence': self._calculate_rank_persistence(i, solution, previous_solutions[i])
                }
        
        # Store for next iteration
        self.ranked_solutions_history.append(current_sorted)
        
        return correspondence, current_sorted
    
    def _calculate_rank_persistence(self, rank: int, current: Solution, previous: Solution) -> float:
        """Calculate how well rank is maintained across time periods"""
        # Measure similarity in objective space
        roi_similarity = 1.0 - abs(current.P.ROI - previous.P.ROI) / max(abs(current.P.ROI), abs(previous.P.ROI), 1e-8)
        risk_similarity = 1.0 - abs(current.P.risk - previous.P.risk) / max(abs(current.P.risk), abs(previous.P.risk), 1e-8)
        
        return 0.5 * (roi_similarity + risk_similarity)
```

**Critérios de Aceitação**:
- [ ] Implementar mapeamento de soluções por ranking
- [ ] Manter identidade de soluções através de ciclos
- [ ] Calcular persistência de ranking
- [ ] Integrar com `AnticipatoryLearning`

---

### **EPIC 4: Setup Experimental e Parâmetros**
**Impacto**: Médio | **Complexidade**: Baixa | **Estimativa**: 1 sprint

#### **User Story 4.1: Configurar Parâmetros da Tese**
```python
# Arquivo: src/config/thesis_parameters.py
class ThesisParameters:
    """Parâmetros experimentais conforme especificação da tese"""
    
    # ASMS Parameters (conforme tese)
    POPULATION_SIZE = 20
    GENERATIONS = 30
    MUTATION_RATE = 0.3
    CROSSOVER_RATE = 0.2
    TOURNAMENT_SIZE = 2
    
    # Constraint Parameters
    MIN_CARDINALITY = 5
    MAX_CARDINALITY = 15
    FEASIBILITY_EPSILON = 0.99
    REFERENCE_POINT = (0.2, 0.0)  # (risk, return)
    
    # KF Parameters
    KF_WINDOW_SIZE = 20  # Configurável
    MONTE_CARLO_SIMULATIONS = 1000
    
    # DD Parameters
    DIRICHLET_SCALE_FACTOR = 1.0
    SLIDING_WINDOW_SIZE = 20  # K parameter
    
    # Anticipatory Learning
    PREDICTION_HORIZON = 2  # H parameter
    LEARNING_RATE_COMBINATION = 0.5  # Weight for Equation 7.16 (thesis-only extension; not in IEEE paper)
```

**Critérios de Aceitação**:
- [ ] Criar classe `ThesisParameters` com valores exatos da tese
- [ ] Implementar configuração flexível (tese vs. padrão)
- [ ] Atualizar todos os experimentos para usar configuração
- [ ] Documentar diferenças entre parâmetros da tese e implementação atual
- [ ] Testes de validação dos parâmetros

---

### **EPIC 5: Multi-Horizon Prediction**
**Impacto**: Médio | **Complexidade**: Média | **Estimativa**: 1-2 sprints

#### **User Story 5.1: Implementar Equação 6.10 Completa**
```python
# Arquivo: src/algorithms/multi_horizon_anticipatory.py
class MultiHorizonAnticipatoryLearning:
    def __init__(self, max_horizon: int = 3):
        self.max_horizon = max_horizon
        self.lambda_rates = {}
    
    def apply_anticipatory_learning_rule(self, current_state: np.ndarray, 
                                       predicted_states: List[np.ndarray], 
                                       lambda_rates: List[float]) -> np.ndarray:
        """
        Implement complete Equation 6.10 / paper Eq (14):
        ẑ_t | z_{t+1:t+H-1} = (1 - Σ_{h=1}^{H-1} λ_{t+h}) z_t + Σ_{h=1}^{H-1} λ_{t+h} ẑ_{t+h} | z_t
        """
        lambda_sum = sum(lambda_rates)
        
        # First term: (1 - Σλ) z_t
        anticipatory_state = (1 - lambda_sum) * current_state
        
        # Second term: Σλ ẑ_{t+h}
        for h, (predicted_state, lambda_h) in enumerate(zip(predicted_states, lambda_rates)):
            anticipatory_state += lambda_h * predicted_state
        
        return anticipatory_state
    
    def calculate_multi_horizon_lambda_rates(self, solution: Solution, 
                                           prediction_horizon: int) -> List[float]:
        """
        Calculate λ_{t+h} rates for multiple horizons
        
        Based on Equation 6.6 / paper Eq (13): λ_{t+h} = (1/(H-1)) [1 - H(p_{t,t+h})]
        """
        lambda_rates = []
        
        for h in range(1, prediction_horizon + 1):
            # Calculate TIP for horizon h
            tip = self._calculate_tip_for_horizon(solution, h)
            
            # Calculate binary entropy
            entropy = self._binary_entropy(tip)
            
            # Calculate lambda rate
            lambda_h = (1.0 / (prediction_horizon - 1)) * (1.0 - entropy)
            lambda_rates.append(max(0.1, min(0.9, lambda_h)))
        
        return lambda_rates
```

**Critérios de Aceitação**:
- [ ] Implementar Equação 6.10 completa
- [ ] Suporte a múltiplos horizontes (H > 1)
- [ ] Cálculo de taxas λ para cada horizonte
- [ ] Integração com `NStepPredictor` existente

---

### **EPIC 4: Belief Coefficient Self-Adjustment**
**Impacto**: Médio | **Complexidade**: Baixa | **Estimativa**: 1 sprint

#### **User Story 4.1: Implementar Equação 6.30**
```python
# Arquivo: src/algorithms/belief_coefficient.py
class BeliefCoefficientCalculator:
    def __init__(self):
        self.tip_history = []
    
    def calculate_belief_coefficient(self, solution: Solution, 
                                   predicted_solution: Solution) -> float:
        """
        Implement Equation 6.30 / paper Eq (20): v_{t+1} = 1 - (1/2) H(p_{t-1,t})
        
        Where H(p_{t-1,t}) is binary entropy of TIP
        """
        # Calculate TIP (Trend Information Probability)
        tip = self._calculate_tip(solution, predicted_solution)
        
        # Calculate binary entropy
        entropy = self._binary_entropy(tip)
        
        # Calculate belief coefficient
        v_t_plus_1 = 1.0 - 0.5 * entropy
        
        # Store for historical analysis
        self.tip_history.append(tip)
        
        return max(0.5, min(1.0, v_t_plus_1))
    
    def _calculate_tip(self, current: Solution, predicted: Solution) -> float:
        """
        Calculate Trend Information Probability (TIP)
        
        Based on Definition 6.1: P_{t,t+h} = Pr[z_t || z_{t+h} | z_t]
        """
        # Get current and predicted objectives
        current_roi, current_risk = current.P.ROI, current.P.risk
        predicted_roi, predicted_risk = predicted.P.ROI, predicted.P.risk
        
        # Calculate probability of mutual non-dominance
        # This is a simplified implementation - full version would use
        # Monte Carlo sampling over probability distributions
        
        # Check dominance relationships
        current_dominates = (current_roi > predicted_roi) and (current_risk < predicted_risk)
        predicted_dominates = (predicted_roi > current_roi) and (predicted_risk < current_risk)
        
        # TIP is probability of mutual non-dominance
        if not current_dominates and not predicted_dominates:
            tip = 0.8  # High TIP when mutually non-dominated
        else:
            tip = 0.2  # Low TIP when one dominates the other
        
        return tip
    
    def _binary_entropy(self, p: float) -> float:
        """Calculate binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)"""
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
```

**Critérios de Aceitação**:
- [ ] Implementar Equação 6.30
- [ ] Cálculo de TIP (Trend Information Probability)
- [ ] Função de entropia binária
- [ ] Integração com sistema de aprendizado existente

---

### **EPIC 5: Enhanced N-Step Prediction Integration**
**Impacto**: Médio | **Complexidade**: Baixa | **Estimativa**: 1 sprint

#### **User Story 5.1: Integrar N-Step com Anticipatory Learning**
```python
# Arquivo: src/algorithms/enhanced_n_step_prediction.py
class EnhancedNStepPredictor(NStepPredictor):
    def __init__(self, max_horizon: int = 3):
        super().__init__(max_horizon)
        self.anticipatory_learning = None
        self.belief_calculator = BeliefCoefficientCalculator()
    
    def set_anticipatory_learning(self, anticipatory_learning):
        """Set reference to anticipatory learning system"""
        self.anticipatory_learning = anticipatory_learning
    
    def compute_conditional_expected_hypervolume(self, pareto_frontier: List[Solution],
                                               selected_solution: int,
                                               kalman_predictions: Dict,
                                               dirichlet_predictions: Dict,
                                               h: int) -> Dict:
        """
        Enhanced version with proper anticipatory learning integration
        
        Based on Pseudocode 7: Anticipatory Distribution Estimation
        """
        if selected_solution >= len(pareto_frontier):
            raise ValueError(f"Invalid solution index {selected_solution}")
        
        # Get the selected solution
        selected = pareto_frontier[selected_solution]
        
        # Get predictions for horizon h
        kalman_pred = kalman_predictions[f'step_{h}']
        dirichlet_pred = dirichlet_predictions[f'step_{h}']
        
        # Calculate belief coefficient for this solution
        belief_coeff = self.belief_calculator.calculate_belief_coefficient(
            selected, selected  # Using same solution for current and predicted
        )
        
        # Compute conditional expectations with belief coefficient
        conditional_hypervolumes = {}
        
        for i, solution in enumerate(pareto_frontier):
            if i == selected_solution:
                # For selected solution, use full expected hypervolume with belief coefficient
                conditional_hv = self._compute_solution_expected_hypervolume(
                    solution, kalman_pred['state'], dirichlet_pred['mean_prediction'], h
                ) * belief_coeff
            else:
                # For other solutions, adjust based on selection
                base_hv = self._compute_solution_expected_hypervolume(
                    solution, kalman_pred['state'], dirichlet_pred['mean_prediction'], h
                )
                
                # Reduce hypervolume due to selection of another solution
                reduction_factor = 0.8 * belief_coeff  # Belief coefficient affects reduction
                conditional_hv = base_hv * reduction_factor
            
            conditional_hypervolumes[f'solution_{i}'] = {
                'conditional_expected_hypervolume': conditional_hv,
                'is_selected': (i == selected_solution),
                'horizon': h,
                'belief_coefficient': belief_coeff
            }
        
        return conditional_hypervolumes
```

**Critérios de Aceitação**:
- [ ] Integrar N-Step com Anticipatory Learning
- [ ] Implementar hypervolume condicional esperado
- [ ] Usar belief coefficient na seleção
- [ ] Testes de integração

---

## 🚀 **PRIORIDADE 2: Melhorias Avançadas (Opcional)**

### **EPIC 6: Advanced Theoretical Components**
**Impacto**: Baixo | **Complexidade**: Alta | **Estimativa**: 2-3 sprints

#### **User Story 6.1: Implementar Pseudocode 6 Completo**
```python
# Arquivo: src/algorithms/dirichlet_map_tracking.py
class DirichletMAPTracking:
    """
    Implement Pseudocode 6: Dirichlet MAP Tracking and Prediction
    
    Complete implementation of the theoretical framework
    """
    def __init__(self, window_size_K: int, concentration_scaling_s: float):
        self.K = window_size_K
        self.s = concentration_scaling_s
        self.sliding_window = SlidingWindowDirichlet(window_size_K, concentration_scaling_s)
    
    def dd_prediction(self, H: int, s: float, u_t: np.ndarray, 
                     U_t_minus_K_to_t_minus_1: List[np.ndarray]) -> np.ndarray:
        """
        Implement Pseudocode 6: DDPREDICTION procedure
        
        Input: H (prediction horizon), s (concentration scaling), 
               u_t (current solution), U_{t-K:t-1} (historical solutions)
        Output: û_{t+H} | u_{t-1} ~ D(s Σ_{k=1}^K m̂_{u_{t+H-k}})
        """
        # Historical DD MAP mean tracking
        for k in range(self.K, 0, -1):
            # Estimate belief coefficient v_{t-k} using Eq. (6.30) / paper Eq (20)
            v_t_minus_k = self._estimate_belief_coefficient(k)
            
            # Predict m̂_{u_{t-k}} using Eq. (6.31) / paper Eq (21)
            m_hat = self._predict_mean_vector(k, U_t_minus_K_to_t_minus_1)
            
            # Update m̂_{u_{t-k}} | m_{u_{t-k}} using Eq. (6.33) / paper Eq (22)
            m_hat_updated = self._map_update(m_hat, U_t_minus_K_to_t_minus_1[k-1])
        
        # H steps ahead prediction
        for h in range(1, H + 1):
            # Estimate belief coefficient v_{t+h} using Eq. (6.30) / paper Eq (20)
            v_t_plus_h = self._estimate_belief_coefficient(h)
            
            # Predict m̂_{u_{t+h}} using Eq. (6.31) / paper Eq (21)
            m_hat_future = self._predict_mean_vector(h, U_t_minus_K_to_t_minus_1)
        
        # Return predicted solution from Dirichlet distribution
        # û_{t+H} | u_{t-1} ~ D(s Σ_{k=1}^K m̂_{u_{t+H-k}})
        predicted_solution = self._sample_from_dirichlet(m_hat_future, s)
        
        return predicted_solution
```

#### **User Story 6.2: Implementar Pseudocode 7 Completo**
```python
# Arquivo: src/algorithms/anticipatory_distribution_estimation.py
class AnticipatoryDistributionEstimation:
    """
    Implement Pseudocode 7: Anticipatory Distribution Estimation
    
    Complete implementation of the theoretical framework
    """
    def __init__(self, max_horizon: int = 3):
        self.max_horizon = max_horizon
        self.dirichlet_tracking = None
        self.kalman_filter = None
    
    def anticipatory_distribution(self, H: int, s: float, u_hat: Solution,
                                U_t: List[Solution], U_hat_t_minus_K_to_t: List[Solution],
                                X_t_minus_K_to_t_minus_1: List[np.ndarray]) -> Solution:
        """
        Implement Pseudocode 7: ANTICIPATORYDISTRIBUTION procedure
        
        Input: H (anticipation horizon), s (concentration scaling),
               u_hat (candidate decision), U_t (current solutions),
               U_hat_{t-K:t}^N (past candidate decisions),
               X_{t-K:t-1} (past observations)
        Output: z_t^(i) | z_{t+1:t+H-1}^(i) (anticipatory distribution)
        """
        # Compute the rank i of u_hat w.r.t. U_t, yielding u_hat^(i)
        rank_i = self._compute_rank(u_hat, U_t)
        u_hat_i = u_hat
        u_hat_i.rank = rank_i
        
        # Set Z_{t:t+H-1}^(i) ← ∅
        Z_t_to_t_plus_H_minus_1 = []
        
        # Regime-specific prediction logic
        if self._is_tlf_regime():
            # TLF Regime: Time-Lagged Feedback
            for h in range(H):
                # z_{t+h}^(i) ← KFPREDICTION(h, u_hat_t^(i), X_{t-K:t-1})
                z_t_plus_h = self._kf_prediction(h, u_hat_i, X_t_minus_K_to_t_minus_1)
                Z_t_to_t_plus_H_minus_1.append(z_t_plus_h)
        else:
            # TL Regime: Time-Lagged
            # Retrieve rank i decisions in U_hat_{t-K:t}^(i) ⊂ U_hat_{t-K:t}
            U_hat_ranked = self._retrieve_ranked_decisions(rank_i, U_hat_t_minus_K_to_t)
            
            for h in range(H):
                # Predict u_hat_{t+h}^(i) using U_hat_{t-K:t}^(i) with Eqs. 6.31-6.33
                u_hat_predicted = self._predict_decision_vector(h, U_hat_ranked)
                
                # Predict z_{t+h}^(i) using Z_{t-K:t}^(i) with the KF
                z_t_plus_h = self._kf_prediction(h, u_hat_predicted, X_t_minus_K_to_t_minus_1)
                
                # z_{t+h}^(i) ← z_{t+h}^(i) + h(m_{u_hat_{t+h}}^(i), m_{u_hat_{t+h-1}}^(i))
                z_t_plus_h = self._add_cost_component(z_t_plus_h, u_hat_predicted, h)
                
                Z_t_to_t_plus_H_minus_1.append(z_t_plus_h)
        
        # Apply OAL over Z_{t:t+H-1}^(i) using Eq. (6.10) / paper Eq (14)
        anticipatory_distribution = self._apply_oal(Z_t_to_t_plus_H_minus_1)
        
        return anticipatory_distribution
```

---

## 📊 **Plano de Implementação**

### **Sprint 1-2: Componentes Críticos**
- [ ] **Sprint 1**: EPIC 1 (Sliding Window Dirichlet) + EPIC 2 (Integração TIP)
- [ ] **Sprint 2**: EPIC 3 (Correspondence Mapping) + EPIC 4 (Setup Experimental)

### **Sprint 3: Melhorias Avançadas**
- [ ] **Sprint 3**: EPIC 5 (Multi-Horizon Prediction) + Testes de Integração
- [ ] Validação contra implementação C++
- [ ] Testes de performance

### **Sprint 4-5: Melhorias Avançadas (Opcional)**
- [ ] **Sprint 4**: EPIC 6 (Belief Coefficient Self-Adjustment)
- [ ] **Sprint 5**: EPIC 7 (Enhanced N-Step Prediction)

---

## 🎯 **Critérios de Sucesso para 100% Aderência**

### **Critérios Técnicos**
- [ ] **100% das equações principais implementadas** (6.6, 6.10, 6.24-6.27, 6.28, 6.30, 6.33, 7.16)
- [ ] **100% dos pseudocódigos implementados** (Pseudocode 5, 6 e 7)
- [ ] **100% dos componentes teóricos** (TIP, belief coefficient, correspondence mapping, constraint handling)
- [ ] **100% de parâmetros experimentais** alinhados com a tese
- [ ] **100% de cobertura de testes** para novos componentes
- [ ] **100% de validação** contra implementação C++

### **Critérios de Qualidade**
- [ ] **Performance**: Não degradar performance existente
- [ ] **Manutenibilidade**: Código bem documentado e modular
- [ ] **Compatibilidade**: Integração perfeita com sistema existente
- [ ] **Robustez**: Tratamento adequado de casos extremos

### **Critérios de Validação**
- [ ] **Testes unitários**: 100% de cobertura para novos componentes
- [ ] **Testes de integração**: Validação end-to-end
- [ ] **Testes de regressão**: Não quebrar funcionalidade existente
- [ ] **Benchmarking**: Comparação com implementação C++

---

## 📈 **Métricas de Progresso**

| Componente | Status Atual | Meta | Progresso |
|------------|--------------|------|-----------|
| Sliding Window Dirichlet | 0% | 100% | 0/100 |
| Correspondence Mapping | 30% | 100% | 30/100 |
| Multi-Horizon Prediction | 40% | 100% | 40/100 |
| Belief Coefficient | 60% | 100% | 60/100 |
| Enhanced N-Step | 50% | 100% | 50/100 |
| **TOTAL** | **80%** | **100%** | **80/100** |

---

## 🚀 **Próximos Passos Imediatos**

1. **Priorizar EPIC 1**: Implementar Sliding Window Dirichlet (maior impacto)
2. **Configurar ambiente de desenvolvimento**: Setup de testes e CI/CD
3. **Criar branch de desenvolvimento**: `feature/100-percent-adherence`
4. **Implementar User Story 1.1**: Equações 6.24-6.27
5. **Validação incremental**: Testes após cada componente

**Estimativa Total**: 4-6 sprints (8-12 semanas) para 100% de aderência
