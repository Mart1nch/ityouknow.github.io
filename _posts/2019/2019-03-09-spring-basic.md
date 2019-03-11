---
layout: post
title: spring简易入门
category: spring
tags: [spring]
---

想直接上手代码的可以先看第二部分。

## 理论基础

Spring是一个轻量级、低侵入性的java ee开源框架。这句话都知道，那么我们先来科普一下。

1、什么是轻量级？

轻量级框架是指那些不需要依赖容器就能运行的框架，比方说都知道EJB是个重量级框架，需要依赖于JBoss等容器；而Spring，Mybatis等都是不需要依赖容器的轻量级框架。

2、什么是侵入性？

当我们需要实现框架的某个功能时，需要我们编写以继承框架某个类的实现方式即说明了该框架的侵入性；举个可能不太合适的栗子，比方说我们都知道，Servlet中监听器需要我们封装一个类继承xxxListener类才能实现监听功能；而Spring中则不需要，所有的功能都可以通过修改xml配置文件实现，这就是低侵入性。

那么Spring有什么功能呢？

简单来说，众所周知的就是aop，ioc，di（等我说完）等

ioc：控制反转（invesion of control）

简单来说就是在大部分情况下，你不再需要自己new一个对象出来了，你只管使用这个对象实例即可，new的操作交给Spring框架。这有什么用呢？我自己new一个对象也花不了多少时间是吧？这就要说到一个很重要的名词了---耦合性。我们都知道，比方说有两个类UserDao和User。我们需要在UserDao中new一个User的实例对象来保存数据，所以我们说UserDao和User这两个类的耦合性是比较高的，UserDao依赖于类User。当User类发生改变，相应的UserDao也需要修改所有与User实例有关的代码。这样给我们维护和修改就带来了麻烦。所以有人提出了ioc的概念，通过引入第三方来实现依赖关系的注入，从而使得两者解耦。

具体可以看一下https://www.cnblogs.com/superjt/p/4311577.html  ，可以说写的很详细了

di：依赖注入（dependency injection）

很多博客都把di和ioc区分的很开，其实这是不对的。di实际上就是ioc的别名。下面划线部分是上面链接中博主的说法。

```
2004年，Martin Fowler（ioc理论的提出者）探讨了同一个问题，既然IOC是控制反转，那么到底是“哪些方面的控制被反转了呢？”，经过详细地分析和论证后，他得出了答案：“获得依赖对象的过程被反转了”。控制被反转之后，获得依赖对象的过程由自身管理变为了由IOC容器主动注入。于是，他给“控制反转”取了一个更合适的名字叫做“依赖注入（Dependency Injection）”。他的这个答案，实际上给出了实现IOC的方法：注入。所谓依赖注入，就是由IOC容器在运行期间，动态地将某种依赖关系注入到对象之中。

所以，依赖注入(DI)和控制反转(IOC)是从不同的角度的描述的同一件事情，就是指通过引入IOC容器，利用依赖关系注入的方式，实现对象之间的解耦。
```

你也可以认为，ioc只是一个理论的方法，而di是其具体实现。当然，di不仅能注入对象实例，也可以注入实例的属性。

说了这么多，如果你知道设计模式的话，你会发现，ioc做的似乎和工厂模式差不多。工厂模式也可以让我们不使用new实例化对象，也可以解耦，我们用spring只是为了方便吗？因为工厂模式内部把实例化某个对象的过程写死了，当我们的需求发生改变的时候需要重新编写、编译工厂类，甚至需要新增工厂类；而ioc的反射机制实例化对象的过程是动态的，不需要重新编译。

aop：面向切面编程（aspect oriented programming）

什么是面向切面编程呢？举个例子，Servlet中的过滤器Filter。在java web项目中页面经常会出现中文乱码的问题，有一种操作简单的方法就是创建一个filter，拦截url后对每个页面先重新设置编码方式“utf-8”之后再解析执行jsp页面中的代码。filter的设计其实就体现了面向切面编程的思想。

## 实战

新建一个SpringDemo，我的目录结构是这样的（下文中没有用到Student的相关内容，故Student及其相关类可以不创建），可以看到导入了哪些包。junit是测试框架，hamcrest是junit依赖的，但是4.1之后的junit不再主动包含hamcrest，需要自己选择合适的版本导入；log4j和commons-logging是日志框架；spring-beans/context/core/expression是实现ioc操作的包；spring-aop是实现aop功能的包。


另：代码全部上传到github了：https://github.com/Mart1nch/HelloSpring，你可以直接clone下来看一下，就不用自己写了


![](http://www.itmind.net/assets/images/2019/springbasic_1.jpg)

 User.java
 
``` java
 package com.bean;

public class User {
	@Override
        // 我们希望print(user)的时候能打印出实例的id，所以加上super.tostring
	public String toString() {
		return "User [name=" + name + ", age=" + age + "]" + "@@" + super.toString();
	}

	private String name;
	private int age;

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getAge() {
		return age;
	}

	public void setAge(int age) {
		this.age = age;
	}

	public User(String name, int age) {
		super();
		this.name = name;
		this.age = age;
	}

	public User() {
	}

}
```
 
 UserDao
 
``` java
package com.dao;

public interface UserDao {
	public void add();

	public void update();
}

```

UserDaoImpl.java
这里注意，Spring注入对象实例有两种注入方式，无参构造方法注入和set方法注入；因为UserDaoImp中需要用到User类 模拟 数据库查找过程，所以需要加上set方法（这里get方法其实用不到）。

``` java
package com.dao.impl;

import com.bean.User;
import com.dao.UserDao;

public class UserDaoImpl implements UserDao {

	private User user;
	
	public User getUser() {
		return user;
	}

	public void setUser(User user) {
		this.user = user;
	}

	@Override
	public void add() {
		System.out.println("UserDaoImpl-add");
	}

	@Override
	public void update() {
		System.out.println("UserDaoImpl-update");
	}
	
        // 模拟了数据库查询返回结果
	public User search() {
		user.setName("mt");
		user.setAge(22);
		return user;
	}

}
```

LoginService.java

同理，这里需要用到User和UserDaoImpl的实例

``` java
package com.service;

import com.bean.User;
import com.dao.UserDao;
import com.dao.impl.UserDaoImpl;

public class LoginService {

	private User user;
	private UserDaoImpl userDaoImpl;
	
	public User getUser() {
		return user;
	}

	public void setUser(User user) {
		this.user = user;
	}

	public UserDaoImpl getUserDaoImpl() {
		return userDaoImpl;
	}

	public void setUserDaoImpl(UserDaoImpl userDaoImpl) {
		this.userDaoImpl = userDaoImpl;
	}

	public void login(User u) {
		user = userDaoImpl.search();

                #这两个打印是为了测试spring bean配置中的singlton模式和prototype模式
		System.out.println(user+"@@"+super.toString());
		System.out.println(u+"@@"+super.toString());
		if(u.getName().equals(user.getName()) && u.getAge()==user.getAge()) {
			System.out.println("login sucessfully");
		}else {
			System.out.println("login failed");
		}
	}
	

}
```

Test1.java

``` java
package com.test;

import org.junit.Test;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import com.bean.User;
import com.dao.impl.UserDaoImpl;
import com.service.LoginService;

public class Test1 {

		User user;
		UserDaoImpl userDaoImpl;
		LoginService loginService;
	
	@Test
	public void t1() {
		ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext.xml");
		user = (User) context.getBean("user");
		userDaoImpl = (UserDaoImpl) context.getBean("userDaoImpl");
		loginService = (LoginService) context.getBean("loginService");
		
		user.setName("mt");
		user.setAge(22);
		loginService.login(user);
	}
	
}
```

applicationContext.xml

官方表示这个spring的配置文件推荐这样命名并且直接保存在src下，否则可能会出问题

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd>

    <!-- scope默认是singleton即单例模式，我这里试了一下prototype，这时在LoginService类中的那两个打印的实例id是不同的 -->
    <bean id="user" class="com.bean.User" scope="prototype"></bean>
    <!-- name value是注入属性，ref是注入配置文件中的bean名称 -->
    <bean id="userDaoImpl" class="com.dao.impl.UserDaoImpl">
      <property name="user" ref="user"></property>
    </bean>
    <bean id="loginService" class="com.service.LoginService">
      <property name="user" ref="user"></property>
      <property name="userDaoImpl" ref="userDaoImpl"></property>
    </bean>
</beans>
```

log4j.properties

```
log4j.rootLogger=DEBUG,Console
log4j.appender.Console=org.apache.log4j.ConsoleAppender
log4j.appender.Console.layout=org.apache.log4j.PatternLayout
log4j.appender.Console.layout.ConversionPattern=%d [%t] %-5p [%c] - %m%n
log4j.logger.org.apache=INFO
```

输出结果：

当applicationContext中配置user的scope是prototype的时候输出是这样的，很明显看到两个user实例的id不同

```
User [name=mt, age=22]@@com.bean.User@4eb7f003
User [name=mt, age=12]@@com.bean.User@eafc191
false
login failed
```

当applicationContext中配置user的scope是singleton的时候，两个实例id是一致的，说明只创建了一个实例

```
User [name=mt, age=22]@@com.bean.User@4eb7f003
User [name=mt, age=22]@@com.bean.User@4eb7f003
true
login sucessfully
```

另外，其实这个spring配置文件中loginService，注入了两个属性，但其实第一个user可以不用注入，因为第二个userDaoImpl中也会再注入一次user，即下面这样也是ok的

``` xml
<bean id="loginService" class="com.service.LoginService">
     <!--  <property name="user" ref="user"></property> -->
      <property name="userDaoImpl" ref="userDaoImpl"></property>
</bean>
```
