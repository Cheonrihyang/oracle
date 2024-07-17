package com.spring.biz.user.impl;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import com.spring.biz.user.UserVO;

@Service("userService")
public class UserServiceImpl implements UserService{

	@Autowired
	@Qualifier("userDAOSpring")
	private UserDAOSpring userDAO;
	
	//set메서드로 구현
	/*
	public void setUserDAO(UserDAO userDAO) {
		this.userDAO = userDAO;
	}
	*/
	
	@Override
	public void insertUser(UserVO vo) {
		System.out.println("[insertUser]메서드 실행");
		userDAO.insertUser(vo);
	}

	@Override
	public void updateUser(UserVO vo) {
		System.out.println("[updateUser]메서드 실행");
		userDAO.updateUser(vo);
		
	}

	@Override
	public void deleteUser(UserVO vo) {
		System.out.println("[deleteUser]메서드 실행");
		userDAO.deleteUser(vo);
		
	}

	@Override
	public UserVO getUser(UserVO vo) {
		System.out.println("[getUser]메서드 실행");
		return userDAO.getUser(vo);
	}

	@Override
	public List<UserVO> getUserList() {
		System.out.println("[getUserList]메서드 실행");
		return userDAO.getUserList();
	}

	@Override
	public UserVO checkUserLogin(UserVO vo) {
		return userDAO.checkUserLogin(vo);
	}



}
