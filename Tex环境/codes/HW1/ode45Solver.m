function [t, S, T] = ode45_solver(a1, a2, a3, a4, S0, T0, t_end)
% 使用ode45求解捕食者-猎物模型
% 输入参数同上，步长由ode45自适应控制

% 定义微分方程右端函数
odefun = @(t, y) [a1 * y(1) * y(2) - a2 * y(1);
                  a3 * y(2) - a4 * y(1) * y(2)];

% 初始条件
y0 = [S0; T0];

% 调用ode45
[t, y] = ode45(odefun, [0, t_end], y0);

% 提取结果
S = y(:, 1);
T = y(:, 2);

end