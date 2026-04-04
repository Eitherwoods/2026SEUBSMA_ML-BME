% 参数设置
a1 = 0.015; a2 = 0.7; a3 = 0.5; a4 = 0.01;
S0 = 100; T0 = 100;
t_end = 100;
dt_list = [0.001, 0.1];

% 循环两种步长
for dt = dt_list
    % 1. Euler法
    [t_e, S_e, T_e] = EulerMethod(a1, a2, a3, a4, S0, T0, t_end, dt);
    
    % 2. 自编RK4法
    [t_rk, S_rk, T_rk] = rk4Method(a1, a2, a3, a4, S0, T0, t_end, dt);
    
    % 3. ode45法（步长自适应，无需指定dt）
    [t_45, S_45, T_45] = ode45Solver(a1, a2, a3, a4, S0, T0, t_end);
    % 使用驼峰命名是因为Latex下划线的问题，来不及修bug了目前
    % 绘图比较（略）
    figure;
    subplot(1,3,1); plot(t_e, S_e, 'b', t_e, T_e, 'r'); title(['Euler, dt=', num2str(dt)]);
    subplot(1,3,2); plot(t_rk, S_rk, 'b', t_rk, T_rk, 'r'); title(['RK4, dt=', num2str(dt)]);
    subplot(1,3,3); plot(t_45, S_45, 'b', t_45, T_45, 'r'); title('ode45');
    sgtitle(['步长 ', num2str(dt), ' 年']);
end