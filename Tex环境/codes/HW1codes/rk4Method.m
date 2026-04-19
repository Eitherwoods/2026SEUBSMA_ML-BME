function [t, S, T] = rk4_method(a1, a2, a3, a4, S0, T0, t_end, dt)
% 四阶Runge-Kutta法求解捕食者-猎物模型
% 输入：
%   a1,a2,a3,a4 - 模型参数
%   S0, T0      - 初始数量
%   t_end       - 终止时间
%   dt          - 时间步长
% 输出：
%   t           - 时间向量
%   S, T        - 鲨鱼和金枪鱼数量向量

N = ceil(t_end / dt);
t = linspace(0, t_end, N+1)';
S = zeros(N+1, 1);
T = zeros(N+1, 1);
S(1) = S0;
T(1) = T0;

for i = 1:N
    % 当前状态
    s = S(i);
    t_curr = T(i);
    
    % 定义右端函数（内联）
    fS = @(s,t) a1*s*t - a2*s;
    fT = @(s,t) a3*t - a4*s*t;
    
    % RK4 斜率
    k1S = fS(s, t_curr);
    k1T = fT(s, t_curr);
    
    k2S = fS(s + dt/2 * k1S, t_curr + dt/2 * k1T);
    k2T = fT(s + dt/2 * k1S, t_curr + dt/2 * k1T);
    
    k3S = fS(s + dt/2 * k2S, t_curr + dt/2 * k2T);
    k3T = fT(s + dt/2 * k2S, t_curr + dt/2 * k2T);
    
    k4S = fS(s + dt * k3S, t_curr + dt * k3T);
    k4T = fT(s + dt * k3S, t_curr + dt * k3T);
    
    % 更新
    S(i+1) = s + dt/6 * (k1S + 2*k2S + 2*k3S + k4S);
    T(i+1) = t_curr + dt/6 * (k1T + 2*k2T + 2*k3T + k4T);
end

end