package com.spring.biz.user.impl;

import java.sql.ResultSet;
import java.sql.SQLException;

import org.springframework.jdbc.core.RowMapper;

import com.spring.biz.user.UserVO;

public class UserMapper implements RowMapper<UserVO>{

	@Override
	public UserVO mapRow(ResultSet rs, int rowNum) throws SQLException {
		UserVO user = new UserVO();
		user.setId(rs.getString("id"));
		user.setPassword(rs.getString("password"));
		user.setName(rs.getString("name"));
		user.setRole(rs.getString("role"));
		return user;
	}

}
