% 主脚本 main.m
clear; clc;
% 参数设置
k21 = 0.5; k23 = 0.1; k32 = 0.4; k02 = 0.2; k03 = 0.3;
f10 = 1;
% 初始条件
x0 = [0; 10; 0];
% 时间跨度
tspan = [0,5];
% 求解 ODE
[t, x] = ode45(@(t,x) drug_3comp(t,x,k21,k23,k32,k02,k03,f10), tspan, x0);
% 绘图
figure;
plot(t, x(:,1), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x(:,2), 'r--', 'LineWidth', 1.5);
plot(t, x(:,3), 'g-.', 'LineWidth', 1.5);
xlabel('时间 t'); ylabel('药物浓度');
legend('房室1', '房室2', '房室3', 'Location', 'best');
title('三房室药物浓度变化曲线');
grid on;
hold off;