function [t, S, T] = euler_method(a1, a2, a3, a4, S0, T0, t_end, dt)
% 显式Euler法求解捕食者-猎物模型
% 输入：
%   a1,a2,a3,a4 - 模型参数
%   S0, T0      - 初始数量
%   t_end       - 终止时间
%   dt          - 时间步长
% 输出：
%   t           - 时间向量
%   S, T        - 鲨鱼和金枪鱼数量向量

N = ceil(t_end / dt);          % 总步数
t = linspace(0, t_end, N+1)';
S = zeros(N+1, 1);
T = zeros(N+1, 1);
S(1) = S0;
T(1) = T0;

for i = 1:N
    S(i+1) = S(i) + dt * (a1 * S(i) * T(i) - a2 * S(i));
    T(i+1) = T(i) + dt * (a3 * T(i) - a4 * S(i) * T(i));
end

end